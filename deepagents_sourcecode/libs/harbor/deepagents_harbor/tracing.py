"""Harbor DeepAgents용 LangSmith 연동 코드입니다."""

import hashlib
import uuid


def create_example_id_from_instruction(instruction: str, seed: int = 42) -> str:
    """instruction 문자열로부터 결정적인(deterministic) UUID를 생성합니다.

    Normalizes the instruction by stripping whitespace and creating a
    SHA-256 hash, then converting to a UUID for LangSmith compatibility.

    Args:
        instruction: The task instruction string to hash
        seed: Integer seed to avoid collisions with existing examples

    Returns:
        A UUID string generated from the hash of the normalized instruction
    """
    # instruction 정규화: 앞/뒤 공백 제거
    normalized = instruction.strip()

    # 해싱을 위해 seed를 bytes로 변환해 instruction 앞에 붙입니다.
    seeded_data = seed.to_bytes(8, byteorder="big") + normalized.encode("utf-8")

    # seed가 포함된 instruction의 SHA-256 해시 생성
    hash_bytes = hashlib.sha256(seeded_data).digest()

    # 앞 16바이트를 사용해 UUID 생성
    example_uuid = uuid.UUID(bytes=hash_bytes[:16])

    return str(example_uuid)
