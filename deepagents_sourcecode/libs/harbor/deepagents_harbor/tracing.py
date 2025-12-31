"""Harbor DeepAgents를 위한 LangSmith 통합."""

import hashlib
import uuid


def create_example_id_from_instruction(instruction: str, seed: int = 42) -> str:
    """지침(instruction) 문자열에서 결정론적(deterministic) UUID를 생성합니다.

    지침을 정규화(앞뒤 공백 제거)하고 SHA-256 해시를 생성한 다음,
    LangSmith 호환성을 위해 UUID로 변환합니다.

    Args:
        instruction: 해시할 작업 지침 문자열
        seed: 기존 예제와의 충돌을 피하기 위한 정수 시드

    Returns:
        정규화된 지침의 해시에서 생성된 UUID 문자열
    """
    # Normalize the instruction: strip leading/trailing whitespace
    normalized = instruction.strip()

    # Prepend seed as bytes to the instruction for hashing
    seeded_data = seed.to_bytes(8, byteorder="big") + normalized.encode("utf-8")

    # Create SHA-256 hash of the seeded instruction
    hash_bytes = hashlib.sha256(seeded_data).digest()

    # Use first 16 bytes to create a UUID
    example_uuid = uuid.UUID(bytes=hash_bytes[:16])

    return str(example_uuid)
