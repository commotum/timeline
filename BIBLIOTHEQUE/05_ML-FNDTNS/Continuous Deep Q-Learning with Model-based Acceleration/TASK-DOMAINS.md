# Continuous Deep Q-Learning with Model-based Acceleration (Not specified in the paper.)
Source: Continuous Deep Q-Learning with Model-based Acceleration.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (cart-pole swing-up) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (reacher target reaching) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (peg insertion) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (gripper manipulation) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (mobile gripper manipulation) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (canada2d ball hitting) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (cheetah locomotion) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (swimmer6 locomotion) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (ant locomotion) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Control (walker2d locomotion) | state of the system | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates continuous control policies across simulated robotic manipulation and locomotion tasks, including cartpole, reacher, peg insertion, gripper variants, canada2d, cheetah, swimmer6, ant, and walker2d.
Inputs are described as the system state, and outputs are actions chosen by the policy.
From this formulation, the tasks are best characterized as operating over fixed-size, non-indexed state/action objects with static attention and direct state (all inferred).

## Evidence

### Task: Control (cart-pole swing-up)
- "Agent must balance a pole attached to a cart by applying forces to the cart alone." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (reacher target reaching)
- "Agent is required to move a 3-DOF arm from random starting locations to random target positions." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (peg insertion)
- "Agent is required to insert the tip of a 3-DOF arm from locally-perturbed starting locations to a fixed hole." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (gripper manipulation)
- "Agent must use an arm with gripper appendage to grasp an object and manuver the object to a fixed target." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (mobile gripper manipulation)
- "Agent must use an arm with gripper attached to a moveable platform to grasp an object and move it to a fixed target." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (canada2d ball hitting)
- "Agent is required to use an arm with hockey-stick like appendage to hit a ball" (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (cheetah locomotion)
- "Agent should move forward as quickly as possible with a cheetah-like body that is constrained to the plane." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (swimmer6 locomotion)
- "Agent should swim in snake-like manner toward the fixed target using six joints, starting from random poses." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (ant locomotion)
- "The four-legged ant should move toward the fixed target from a fixed starting position and posture." (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)

### Task: Control (walker2d locomotion)
- "Agent should move forward as quickly as possible with a bipedal walker constrained to the plane" (Table 3, Section 8.4)
- "the input to the policy consisted of the state of the system" (Section 6, Experiments)
- "the agent chooses an action  $u_t$" (Section 3, Background)
- Inference: Inferred 0D/Fixed input-output, Static attention, and Direct state because the policy uses the system state as input and outputs actions. (Section 6, Experiments; Section 3, Background)
