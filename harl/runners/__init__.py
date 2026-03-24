"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.greedy_rule_base_runner import RuleBaseRunner
from harl.runners.genetic_base_runner import GeneticRunner
from harl.runners.on_policy_mat_runner import OnPolicyMATRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hadt": OnPolicyHARunner,
    "mappo": OnPolicyMARunner,
    "rulebased": RuleBaseRunner,
    "genetic": GeneticRunner,
    "mat": OnPolicyMATRunner,
}
