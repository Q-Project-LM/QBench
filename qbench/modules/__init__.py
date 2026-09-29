from . import agentic, knowledge

MODULE_REGISTRY = {
    "agentic": agentic.run,
    "knowledge": knowledge.run,
}
