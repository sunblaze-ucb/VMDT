from uuid import uuid4


def gen_id(prefix: str = "", suffix: str = ""):
    return prefix + uuid4().hex[:8] + suffix
