import datacreek.utils.crypto as crypto


def test_xor_encrypt_roundtrip():
    token = crypto.xor_encrypt('secret', 'k')
    assert token != 'secret'
    assert crypto.xor_decrypt(token, 'k') == 'secret'


def test_encrypt_decrypt_fields_missing():
    record = {'name': 'Alice', 'ssn': '123', 'age': None}
    crypto.encrypt_pii_fields(record, 'k', ['ssn', 'age', 'missing'])
    assert record['name'] == 'Alice'
    assert record['age'] is None
    encrypted = record['ssn']
    assert encrypted != '123'
    crypto.decrypt_pii_fields(record, 'k', ['ssn', 'age'])
    assert record['ssn'] == '123'
    assert record['age'] is None
