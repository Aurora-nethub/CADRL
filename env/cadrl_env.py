# env/cadrl_env.py
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from core import ConfigContainer  # 可接受你的配置容器
from utils.action import Action
from utils.state import FullState, BaseState, JointState
from sim.dynamics import rollout_state
from sim.reward import compute_reward, RewardSpec

SPEC = RewardSpec()

@dataclass
class _Agent:
    """环境内部用的轻量实体状态（世界坐标系）"""
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    pgx: float
    pgy: float
    v_pref: float
    theta: float
    kinematic: bool
    done: int = 0  # 0 active, 1 reached, 2 collision, 3 out-of-bound, 4 timeout


class CADRLEnv:
    """
    两车 CADRL 环境（世界坐标系）：
      - reset(): 初始化双车的起点/终点
      - step(actions): 先用 sim.reward 计算 (reward, end_time_ratio)，再用
        sim.dynamics 按各自 end_time_ratio 推进，返回新 JointState/奖励/终止
      - peek(agent_idx, action, other_action): 无副作用前瞻（给 Agent 做单步打分）

    说明：
      * value network 的“坐标对齐/旋转”在网络内部做，本环境**不做旋转**。
      * 解析式最近距或三点采样可通过 collision_mode 切换（默认 analytic）。
    """

    def __init__(
        self,
        cfg: ConfigContainer,
        phase: str = "train",
        *,
        collision_mode: str = "analytic",  # "analytic" 或 "sample3"
    ) -> None:
        assert phase in ("train", "test")
        assert collision_mode in ("analytic", "sample3")
        self.cfg = cfg
        self.phase = phase
        # 优先读取cfg.sim.dt，否则用默认0.1
        self.dt = float(getattr(getattr(cfg, "sim", object()), "dt", 0.1))
        self.collision_mode = collision_mode

        # 读取配置
        self.radius = float(cfg.agent.radius)
        self.v_pref = float(cfg.agent.v_pref)
        self.kinematic = bool(cfg.agent.kinematic)

        self.agent_num = int(cfg.sim.agent_num)
        assert self.agent_num == 2, "当前 CADRL 环境仅支持两车"

        self.crossing_radius = float(cfg.sim.crossing_radius)
        self.xmin = float(cfg.sim.xmin)
        self.xmax = float(cfg.sim.xmax)
        self.ymin = float(cfg.sim.ymin)
        self.ymax = float(cfg.sim.ymax)
        self.max_steps = int(cfg.sim.max_time)  # 与原仓库一样，用“步”计数

        # 运行时状态
        self._agents: List[_Agent] = [None, None]  # type: ignore
        self._step_count: int = 0
        self._last_step_ratio: float = 1.0  # 本步推进比例（便于外部累计真实时间）
        self._test_counter: int = 0

        # 扰动范围（可由外部动态设置）
        self.disturb_range: Tuple[float, float] = (0.7, 1.3)

    # ---------------- 公有接口 ----------------


    def reset(self, case: Optional[int] = None, disturb_cr: bool = True) -> List[JointState]:
        """
        初始化两车：自车(-cr,0)->(+cr,0)，对手在半径 cr 上取角度反向
        cr扰动范围由 self.disturb_range 控制，默认训练和测试都扰动（可通过disturb_cr关闭）
        case参数用于测试集复现
        """
        base_cr = self.crossing_radius
        if disturb_cr:
            cr = random.uniform(*self.disturb_range) * base_cr
        else:
            cr = base_cr

        # 自车从左到右
        self._agents[0] = _Agent(
            px=-cr, py=0.0, vx=0.0, vy=0.0, radius=self.radius,
            pgx=cr, pgy=0.0, v_pref=self.v_pref, theta=0.0, kinematic=self.kinematic
        )

        # 对手角度（训练随机、测试均匀分布）
        if self.phase == "train":
            angle = random.random() * math.pi
            # 与开源一致的小间隔约束，避免太近
            while math.sin((math.pi - angle) / 2.0) < 0.3 / 2.0:
                angle = random.random() * math.pi
        else:
            n_cases = 20  # 测试case分辨率，和test.py保持一致
            if case is not None:
                angle = (case % n_cases) / n_cases * 2 * math.pi
                self._test_counter = case
            else:
                angle = (self._test_counter % n_cases) / n_cases * 2 * math.pi
                self._test_counter += 1

        x, y = cr * math.cos(angle), cr * math.sin(angle)
        theta = angle + math.pi  # 面向原点

        self._agents[1] = _Agent(
            px=x, py=y, vx=0.0, vy=0.0, radius=self.radius,
            pgx=-x, pgy=-y, v_pref=self.v_pref, theta=theta, kinematic=self.kinematic
        )

        self._step_count = 0
        self._last_step_ratio = 1.0

        return [self._joint_state(0), self._joint_state(1)]

    def step(self, actions: List[Action]) -> Tuple[List[JointState], List[float], List[int]]:
        """输入两个动作 → (states, rewards, done_signals)"""
        assert len(actions) == 2, "需要两个动作"
        rewards: List[float] = [0.0, 0.0]
        end_ratios: List[float] = [1.0, 1.0]
        done_signals: List[int] = []

        # 1) 奖励与截断时间（两边都用"已知双方动作"的评分）
        # 重要：只对未终止的 agent 计算 reward，已终止的保持 0
        for i in range(2):
            if self._agents[i].done != 0:
                # 已终止的 agent 不再累计 reward
                rewards[i] = 0.0
                end_ratios[i] = 1.0
            else:
                s_i = self._joint_tensor(i)  # (14,) CPU
                r_i, end_i = compute_reward(
                    s_i, actions[i], actions[1 - i],
                    dt=self.dt, kinematic=self.kinematic,
                    collision_mode=self.collision_mode, goal_tolerance=self.radius
                )
                rewards[i] = float(r_i)
                end_ratios[i] = float(end_i)

        # 碰撞互斥（按开源实现，对称）
        # --- 碰撞同步化：任一侧判为碰撞 -> 双方都按碰撞处理 ---
        COLL_PEN = float(getattr(getattr(self.cfg, "reward", object()), "collision_penalty", -0.25))  # pylint: disable=invalid-name
        GOAL_REW = float(getattr(getattr(self.cfg, "reward", object()), "goal_reward", 1.0))  # pylint: disable=invalid-name
        # 数值容差：compute_reward 会在终止步上叠加一个很小的 time_penalty（例如 -0.008*dt ≈ -8e-4），
        # 若直接用奖励值等于 1.0/-0.25 判定，将错过终止信号；放宽 EPS 以兼容这类微小偏移。
        EPS = 1e-3  # pylint: disable=invalid-name

        col0 = abs(rewards[0] - COLL_PEN) < EPS
        col1 = abs(rewards[1] - COLL_PEN) < EPS

        # 碰撞是双方事件，需要同步（任一判为碰撞，双方都按碰撞处理）
        if col0 or col1:
            # 若某一侧判定为碰撞，双方都按碰撞处理
            # 用"更早"的截断比例（谁先撞用谁），保证推进一致
            t_ratio = min(float(end_ratios[0]), float(end_ratios[1]))
            rewards[0] = rewards[1] = COLL_PEN
            end_ratios[0] = end_ratios[1] = t_ratio
        # 注意：goal 是单方事件，不需要同步！
        # 每个智能体的 goal 奖励由 compute_reward 独立计算，此处不覆盖


        # 2) 依据各自 end_time_ratio 推进（用 sim.dynamics，避免重复公式）
        for i in range(2):
            s_i = self._joint_tensor(i)
            eff_dt = self.dt * end_ratios[i]
            s_next = rollout_state(
                s_i, actions[i], actions[1 - i],
                dt=eff_dt, kinematic=self.kinematic,
            )
            # 把“自车视角”的前 9 维落回到世界：更新 agent i
            self._apply_self_from_joint(i, s_next)

        # 3) 组合新的 JointState、更新 done
        states = [self._joint_state(i) for i in range(2)]

        # 计算真实碰撞（当前位置距离）- 与 reward.py 中使用相同的判定逻辑
        dist = math.hypot(
            self._agents[0].px - self._agents[1].px,
            self._agents[0].py - self._agents[1].py
        )
        sum_r = self._agents[0].radius + self._agents[1].radius
        eps_c = 1e-8 * float(max(1.0, sum_r))
        real_collision = (dist <= sum_r + eps_c)

        # 额外：用几何方式判断“到达目标”（避免依赖奖励值的精确相等）
        goal_reached = [False, False]
        for i in range(2):
            a = self._agents[i]
            gdist = math.hypot(a.px - a.pgx, a.py - a.pgy)
            # 使用与半径同量级的容差
            goal_reached[i] = (gdist <= (a.radius + 1e-8))

        # 更新 done 状态（只处理未终止的 agent）
        for i in range(2):
            a = self._agents[i]
            if a.done == 0:  # 只处理活跃的
                # 判定优先级：碰撞 > 到达 > 越界 > 超时
                if real_collision:
                    # 真实碰撞：与 reward 判定使用相同距离
                    a.done = 2
                elif goal_reached[i]:
                    # 到达目标（几何判定）
                    a.done = 1
                elif not self._in_bound(i):
                    # 越界
                    a.done = 3
                elif self._step_count >= self.max_steps:
                    # 超时
                    a.done = 4
                else:
                    a.done = 0
            done_signals.append(a.done)

        self._step_count += 1
        self._last_step_ratio = max(end_ratios)
        return states, rewards, done_signals

    def peek(
        self,
        agent_idx: int,
        action: Action,
        other_action: Optional[Action] = None
    ) -> Tuple[torch.Tensor, float, float]:
        """
        无副作用前瞻（给 Agent 单步打分）：
          返回 (next_joint_state_tensor(14,), reward, end_time_ratio∈[0,1])
        """
        s = self._joint_tensor(agent_idx)
        r, end_ratio = compute_reward(
            s, action, other_action,
            dt=self.dt, kinematic=self.kinematic,
            collision_mode=self.collision_mode, goal_tolerance=self.radius
        )
        s_next = rollout_state(
            s, action, other_action,
            dt=self.dt * float(end_ratio), kinematic=self.kinematic
        )
        return s_next, float(r), float(end_ratio)

    # ---------------- 内部工具 ----------------

    def _joint_state(self, agent_idx: int) -> JointState:
        """世界状态 → JointState 对象（自车完整 + 对手可观测）"""
        me = self._agents[agent_idx]
        ot = self._agents[1 - agent_idx]
        return JointState(
            self_state=FullState(me.px, me.py, me.vx, me.vy, me.radius,
                                 me.pgx, me.pgy, me.v_pref, me.theta),
            neighbor_state=BaseState(ot.px, ot.py, ot.vx, ot.vy, ot.radius),
        )

    def _joint_tensor(self, agent_idx: int) -> torch.Tensor:
        """JointState → (14,) CPU tensor"""
        return self._joint_state(agent_idx).to_tensor(as_batch=False)

    def _apply_self_from_joint(self, agent_idx: int, joint_next: torch.Tensor) -> None:
        """把自车视角的前 9 维状态写回环境中的 agent"""
        a = self._agents[agent_idx]
        a.px = float(joint_next[0].item())
        a.py = float(joint_next[1].item())
        a.vx = float(joint_next[2].item())
        a.vy = float(joint_next[3].item())
        a.theta = float(joint_next[8].item()) if a.kinematic else 0.0
        # 其余（pgx/pgy/v_pref/radius）不变

    def _in_bound(self, agent_idx: int) -> bool:
        a = self._agents[agent_idx]
        return (self.xmin < a.px < self.xmax) and (self.ymin < a.py < self.ymax)

    # ---------------- 便捷属性 ----------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_step_ratio(self) -> float:
        """本步推进比例（∈[0,1]），用于外部按真实时间累计：t_acc += last_step_ratio * dt"""
        return self._last_step_ratio
