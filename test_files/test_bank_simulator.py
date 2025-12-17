import pytest
import json
import os
from datetime import datetime
from decimal import Decimal
from unittest import mock

# Assuming the code to be tested is in Bank_Simulator.py
# For testing, we'll import directly. In a real project, this might be a package install.
# If running this test file directly, ensure Bank_Simulator.py is in the same directory
# or properly importable.

# To handle imports relative to the module under test, sometimes it's necessary to
# add the directory to sys.path, but for a single file, direct import is fine.
from Bank_Simulator import (
    now, money, BankError, NotFound, Invalid, InsufficientFunds,
    Account, Tx, Counter, Bank, load, save, show_account, show_tx,
    parse_amount, repl, main, asdict
)

# Mock datetime.utcnow to make 'now()' deterministic
FIXED_DATETIME_STR = "2023-10-27T10:00:00Z"
FIXED_DATETIME_OBJ = datetime(2023, 10, 27, 10, 0, 0)

@pytest.fixture(autouse=True)
def mock_datetime_now():
    with mock.patch('Bank_Simulator.datetime') as mock_dt:
        mock_dt.utcnow.return_value = FIXED_DATETIME_OBJ
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw) # Allow other datetime calls to work
        yield mock_dt

class TestHelperFunctions:
    def test_now(self):
        assert now() == FIXED_DATETIME_STR

    @pytest.mark.parametrize("value, expected", [
        ("100", Decimal("100.00")),
        (100, Decimal("100.00")),
        (100.5, Decimal("100.50")),
        (Decimal("99.999"), Decimal("100.00")),  # Round half up
        (Decimal("99.991"), Decimal("99.99")),
        (Decimal("123.456"), Decimal("123.46")),
        (Decimal("123.455"), Decimal("123.46")),
        (Decimal("123.454"), Decimal("123.45")),
        ("0", Decimal("0.00")),
        (Decimal("-50.235"), Decimal("-50.23")), # Round half up
        (Decimal("-50.236"), Decimal("-50.24")), # Round half up
    ])
    def test_money(self, value, expected):
        assert money(value) == expected

    def test_parse_amount(self):
        assert parse_amount("100.50") == Decimal("100.50")
        assert parse_amount(" 100 ") == Decimal("100.00")
        with pytest.raises(Invalid, match="amount required"):
            parse_amount("")
        with pytest.raises(Invalid, match="amount required"):
            parse_amount("   ")
        with pytest.raises(Exception): # Decimal conversion can raise various errors
            parse_amount("abc")
        with pytest.raises(Exception):
            parse_amount("100.x")

class TestAccount:
    def test_account_init(self):
        acc = Account("A001", "Alice", "USD")
        assert acc.account_id == "A001"
        assert acc.owner == "Alice"
        assert acc.currency == "USD"
        assert acc.balance == "0.00"
        assert acc.status == "active"
        assert acc.created_at == FIXED_DATETIME_STR

        # With initial balance and custom date
        acc_with_bal = Account("A002", "Bob", "EUR", balance="100.50", created_at="2022-01-01T00:00:00Z")
        assert acc_with_bal.balance == "100.50"
        assert acc_with_bal.created_at == "2022-01-01T00:00:00Z"

    def test_account_bal_set_bal(self):
        acc = Account("A001", "Alice", "USD", balance="50.25")
        assert acc.bal() == Decimal("50.25")

        acc.set_bal(Decimal("100.75"))
        assert acc.balance == "100.75"
        assert acc.bal() == Decimal("100.75")

        acc.set_bal(Decimal("123.456")) # Test rounding
        assert acc.balance == "123.46"
        assert acc.bal() == Decimal("123.46")

class TestTx:
    def test_tx_init(self):
        tx = Tx(
            tx_id="T001",
            kind="deposit",
            amount="100.00",
            currency="USD",
            ts="2023-01-01T12:00:00Z",
            dst="A001",
            note="Test deposit"
        )
        assert tx.tx_id == "T001"
        assert tx.kind == "deposit"
        assert tx.amount == "100.00"
        assert tx.currency == "USD"
        assert tx.ts == "2023-01-01T12:00:00Z"
        assert tx.src is None
        assert tx.dst == "A001"
        assert tx.note == "Test deposit"

        tx_transfer = Tx(
            tx_id="T002",
            kind="transfer",
            amount="50.00",
            currency="EUR",
            ts="2023-01-01T12:01:00Z",
            src="A001",
            dst="A002",
            note="Test transfer"
        )
        assert tx_transfer.src == "A001"
        assert tx_transfer.dst == "A002"

class TestCounter:
    def test_counter_init_and_next(self):
        c = Counter("P")
        assert c.prefix == "P"
        assert c.n == 1
        assert c.next() == "P000001"
        assert c.n == 2
        assert c.next() == "P000002"

        c2 = Counter("Q", 100)
        assert c2.next() == "Q000100"
        assert c2.next() == "Q000101"

    def test_counter_snap_from_snap(self):
        c = Counter("X", 5)
        c.next() # X000005, n becomes 6
        snap = c.snap()
        assert snap == {"prefix": "X", "n": 6}

        c_restored = Counter.from_snap(snap)
        assert c_restored.prefix == "X"
        assert c_restored.n == 6
        assert c_restored.next() == "X000006"
        assert c_restored.n == 7

        # Test with invalid types in snap dict
        with pytest.raises(TypeError):
            Counter.from_snap({"prefix": 123, "n": 4})
        with pytest.raises(TypeError):
            Counter.from_snap({"prefix": "X", "n": "abc"})


class TestBank:
    def test_bank_init(self):
        bank = Bank("TestBank")
        assert bank.name == "TestBank"
        assert not bank.accounts
        assert not bank.txs
        assert bank._acct.prefix == "A"
        assert bank._tx.prefix == "T"

    def test_open_account(self, mock_datetime_now):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice Smith", "USD")
        assert acc.owner == "Alice Smith"
        assert acc.currency == "USD"
        assert acc.account_id == "A000001"
        assert acc.balance == "0.00"
        assert acc.created_at == FIXED_DATETIME_STR
        assert bank.accounts["A000001"] == acc
        assert not bank.txs

        acc2 = bank.open_account("Bob Johnson", "EUR", initial=Decimal("100.50"))
        assert acc2.owner == "Bob Johnson"
        assert acc2.currency == "EUR"
        assert acc2.account_id == "A000002"
        assert acc2.balance == "100.50"
        assert bank.accounts["A000002"] == acc2
        assert len(bank.txs) == 1
        tx = bank.txs[0]
        assert tx.kind == "deposit"
        assert tx.amount == "100.50"
        assert tx.currency == "EUR"
        assert tx.dst == "A000002"
        assert tx.note == "initial"
        assert tx.ts == FIXED_DATETIME_STR
        assert tx.tx_id == "T000001"

        # Test trimming and capitalization
        acc3 = bank.open_account("  Charlie  ", "  gbp  ", initial="50.25")
        assert acc3.owner == "Charlie"
        assert acc3.currency == "GBP"
        assert acc3.balance == "50.25"

    @pytest.mark.parametrize("owner, currency, initial, expected_error", [
        ("", "USD", "100", Invalid("owner required")),
        ("Alice", "", "100", Invalid("currency invalid")),
        ("Alice", "US", "100", Invalid("currency invalid")),
        ("Alice", "USDD", "100", Invalid("currency invalid")),
    ])
    def test_open_account_invalid(self, owner, currency, initial, expected_error):
        bank = Bank("TestBank")
        with pytest.raises(type(expected_error), match=str(expected_error)):
            bank.open_account(owner, currency, initial=Decimal(initial))

    def test_get_account(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD")
        assert bank.get(acc.account_id) == acc
        assert bank.get(f" {acc.account_id} ") == acc # Test trimming

        with pytest.raises(NotFound, match="account not found"):
            bank.get("NonExistent")

    def test_active_account(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD")
        assert bank._active(acc.account_id) == acc

        acc.status = "inactive"
        with pytest.raises(Invalid, match="account not active"):
            bank._active(acc.account_id)

        with pytest.raises(NotFound, match="account not found"):
            bank._active("NonExistent")

    def test_deposit(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD")
        assert acc.bal() == Decimal("0.00")

        tx1 = bank.deposit(acc.account_id, Decimal("100.50"), note="Paycheck")
        assert acc.bal() == Decimal("100.50")
        assert len(bank.txs) == 1
        assert bank.txs[0] == tx1
        assert tx1.kind == "deposit"
        assert tx1.amount == "100.50"
        assert tx1.dst == acc.account_id
        assert tx1.note == "Paycheck"
        assert tx1.tx_id == "T000001"

        tx2 = bank.deposit(acc.account_id, Decimal("49.50"))
        assert acc.bal() == Decimal("150.00")
        assert len(bank.txs) == 2
        assert bank.txs[1] == tx2
        assert tx2.tx_id == "T000002"
        assert tx2.note == ""

    @pytest.mark.parametrize("amount, expected_error", [
        (Decimal("0"), Invalid("amount must be > 0")),
        (Decimal("-10"), Invalid("amount must be > 0")),
    ])
    def test_deposit_invalid_amount(self, amount, expected_error):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD")
        with pytest.raises(type(expected_error), match=str(expected_error)):
            bank.deposit(acc.account_id, amount)

    def test_deposit_non_existent_or_inactive(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD")
        acc.status = "inactive"

        with pytest.raises(NotFound, match="account not found"):
            bank.deposit("NonExistent", Decimal("50"))
        with pytest.raises(Invalid, match="account not active"):
            bank.deposit(acc.account_id, Decimal("50"))

    def test_withdraw(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD", initial=Decimal("200.00"))
        assert acc.bal() == Decimal("200.00")

        tx1 = bank.withdraw(acc.account_id, Decimal("50.25"), note="Groceries")
        assert acc.bal() == Decimal("149.75")
        assert len(bank.txs) == 2 # 1 initial deposit + 1 withdraw
        assert bank.txs[1] == tx1
        assert tx1.kind == "withdraw"
        assert tx1.amount == "50.25"
        assert tx1.src == acc.account_id
        assert tx1.note == "Groceries"
        assert tx1.tx_id == "T000002"

        tx2 = bank.withdraw(acc.account_id, Decimal("49.75"))
        assert acc.bal() == Decimal("100.00")
        assert len(bank.txs) == 3
        assert bank.txs[2] == tx2
        assert tx2.tx_id == "T000003"
        assert tx2.note == ""

    @pytest.mark.parametrize("initial, withdraw_amount, expected_error", [
        (Decimal("100"), Decimal("0"), Invalid("amount must be > 0")),
        (Decimal("100"), Decimal("-10"), Invalid("amount must be > 0")),
        (Decimal("50"), Decimal("100"), InsufficientFunds("insufficient funds")),
        (Decimal("50.00"), Decimal("50.01"), InsufficientFunds("insufficient funds")),
    ])
    def test_withdraw_invalid_amount_or_funds(self, initial, withdraw_amount, expected_error):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD", initial=initial)
        with pytest.raises(type(expected_error), match=str(expected_error)):
            bank.withdraw(acc.account_id, withdraw_amount)

    def test_withdraw_non_existent_or_inactive(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD", initial=Decimal("100"))
        acc.status = "inactive"

        with pytest.raises(NotFound, match="account not found"):
            bank.withdraw("NonExistent", Decimal("50"))
        with pytest.raises(Invalid, match="account not active"):
            bank.withdraw(acc.account_id, Decimal("50"))

    def test_transfer(self):
        bank = Bank("TestBank")
        src_acc = bank.open_account("Alice", "USD", initial=Decimal("200.00"))
        dst_acc = bank.open_account("Bob", "USD", initial=Decimal("50.00"))
        assert src_acc.bal() == Decimal("200.00")
        assert dst_acc.bal() == Decimal("50.00")
        assert len(bank.txs) == 2 # initial deposits

        tx = bank.transfer(src_acc.account_id, dst_acc.account_id, Decimal("75.50"), note="Payment")
        assert src_acc.bal() == Decimal("124.50")
        assert dst_acc.bal() == Decimal("125.50")
        assert len(bank.txs) == 3
        assert bank.txs[2] == tx
        assert tx.kind == "transfer"
        assert tx.amount == "75.50"
        assert tx.currency == "USD"
        assert tx.src == src_acc.account_id
        assert tx.dst == dst_acc.account_id
        assert tx.note == "Payment"
        assert tx.tx_id == "T000003"

    @pytest.mark.parametrize("src_bal, dst_bal, src_ccy, dst_ccy, amount, expected_error", [
        ("100", "0", "USD", "USD", "10", None), # Valid
        ("100", "0", "USD", "USD", "0", Invalid("amount must be > 0")),
        ("100", "0", "USD", "USD", "-10", Invalid("amount must be > 0")),
        ("10", "0", "USD", "USD", "20", InsufficientFunds("insufficient funds")),
        ("100", "0", "USD", "EUR", "10", Invalid("currency mismatch")),
    ])
    def test_transfer_invalid_cases(self, src_bal, dst_bal, src_ccy, dst_ccy, amount, expected_error):
        bank = Bank("TestBank")
        src_acc = bank.open_account("Alice", src_ccy, initial=Decimal(src_bal))
        dst_acc = bank.open_account("Bob", dst_ccy, initial=Decimal(dst_bal))

        if expected_error:
            with pytest.raises(type(expected_error), match=str(expected_error)):
                bank.transfer(src_acc.account_id, dst_acc.account_id, Decimal(amount))
        else:
            bank.transfer(src_acc.account_id, dst_acc.account_id, Decimal(amount))
            assert src_acc.bal() == Decimal(src_bal) - Decimal(amount)
            assert dst_acc.bal() == Decimal(dst_bal) + Decimal(amount)

    def test_transfer_same_account(self):
        bank = Bank("TestBank")
        acc = bank.open_account("Alice", "USD", initial=Decimal("100"))
        with pytest.raises(Invalid, match="same account"):
            bank.transfer(acc.account_id, acc.account_id, Decimal("10"))

    def test_transfer_non_existent_or_inactive_accounts(self):
        bank = Bank("TestBank")
        acc1 = bank.open_account("Alice", "USD", initial=Decimal("100"))
        acc2 = bank.open_account("Bob", "USD", initial=Decimal("100"))

        with pytest.raises(NotFound, match="account not found"):
            bank.transfer("NonExistent", acc2.account_id, Decimal("10"))
        with pytest.raises(NotFound, match="account not found"):
            bank.transfer(acc1.account_id, "NonExistent", Decimal("10"))

        acc1.status = "inactive"
        with pytest.raises(Invalid, match="account not active"):
            bank.transfer(acc1.account_id, acc2.account_id, Decimal("10"))
        acc1.status = "active" # Reset for next test
        acc2.status = "inactive"
        with pytest.raises(Invalid, match="account not active"):
            bank.transfer(acc1.account_id, acc2.account_id, Decimal("10"))

    def test_list_accounts(self):
        bank = Bank("TestBank")
        assert not bank.list_accounts()

        acc1 = bank.open_account("Alice", "USD")
        acc2 = bank.open_account("Bob", "EUR")
        accounts = bank.list_accounts()
        assert len(accounts) == 2
        assert acc1 in accounts
        assert acc2 in accounts

    def test_list_txs(self):
        bank = Bank("TestBank")
        acc1 = bank.open_account("Alice", "USD", initial=Decimal("100")) # T000001
        acc2 = bank.open_account("Bob", "EUR", initial=Decimal("50"))   # T000002
        bank.deposit(acc1.account_id, Decimal("20"), note="Bonus") # T000003
        bank.withdraw(acc2.account_id, Decimal("10"), note="Fee")  # T000004
        bank.transfer(acc1.account_id, acc2.account_id, Decimal("30"), note="Gift") # T000005

        all_txs = bank.list_txs()
        assert len(all_txs) == 5
        # Ensure order is reversed (most recent first)
        assert all_txs[0].tx_id == "T000005"
        assert all_txs[-1].tx_id == "T000001"

        # Test limit
        limited_txs = bank.list_txs(limit=3)
        assert len(limited_txs) == 3
        assert limited_txs[0].tx_id == "T000005"
        assert limited_txs[2].tx_id == "T000003"

        # Test filtering by account_id
        acc1_txs = bank.list_txs(account_id=acc1.account_id)
        assert len(acc1_txs) == 3 # T000001 (init), T000003 (deposit), T000005 (transfer src)
        assert acc1_txs[0].tx_id == "T000005"
        assert acc1_txs[1].tx_id == "T000003"
        assert acc1_txs[2].tx_id == "T000001"

        acc2_txs = bank.list_txs(account_id=acc2.account_id, limit=1)
        assert len(acc2_txs) == 1 # T000005 (transfer dst) should be first when reversed
        assert acc2_txs[0].tx_id == "T000005"

        # Test filtering with limit
        acc1_limited_txs = bank.list_txs(account_id=acc1.account_id, limit=2)
        assert len(acc1_limited_txs) == 2
        assert acc1_limited_txs[0].tx_id == "T000005"
        assert acc1_limited_txs[1].tx_id == "T000003"

        # Test invalid limit
        with pytest.raises(Invalid, match="limit must be > 0"):
            bank.list_txs(limit=0)
        with pytest.raises(Invalid, match="limit must be > 0"):
            bank.list_txs(limit=-5)

        # Test non-existent account for filter
        with pytest.raises(NotFound, match="account not found"):
            bank.list_txs(account_id="NonExistent")

    def test_bank_to_from_dict(self):
        bank = Bank("MyAwesomeBank")
        acc1 = bank.open_account("Alice", "USD", initial=Decimal("100.50"))
        acc2 = bank.open_account("Bob", "EUR")
        bank.deposit(acc1.account_id, Decimal("20.00"), note="Gift")
        bank.transfer(acc1.account_id, acc2.account_id, Decimal("10.00"), note="Shared bill")
        bank.withdraw(acc2.account_id, Decimal("5.00"))
        
        # Manually change an account status to ensure it's saved/loaded
        acc1.status = "inactive"
        acc1.balance = "110.50" # Update balance after transactions to match.

        bank_dict = bank.to_dict()
        
        # Verify basic structure
        assert bank_dict["name"] == "MyAwesomeBank"
        assert len(bank_dict["accounts"]) == 2
        assert len(bank_dict["txs"]) == 5 # 2 initial, 3 operations
        assert "counters" in bank_dict
        assert bank_dict["counters"]["acct"]["n"] == 3
        assert bank_dict["counters"]["tx"]["n"] == 6

        # Restore from dict
        restored_bank = Bank.from_dict(bank_dict)

        assert restored_bank.name == bank.name
        assert len(restored_bank.accounts) == len(bank.accounts)
        assert len(restored_bank.txs) == len(bank.txs)

        # Compare accounts
        for aid, acc in bank.accounts.items():
            restored_acc = restored_bank.accounts.get(aid)
            assert restored_acc is not None
            assert asdict(restored_acc) == asdict(acc) # Compare dataclass as dict

        # Compare transactions
        for i, tx in enumerate(bank.txs):
            assert asdict(restored_bank.txs[i]) == asdict(tx)

        # Compare counters
        assert restored_bank._acct.n == bank._acct.n
        assert restored_bank._tx.n == bank._tx.n
        assert restored_bank._acct.prefix == bank._acct.prefix
        assert restored_bank._tx.prefix == bank._tx.prefix

        # Test opening new account after restoration
        new_acc = restored_bank.open_account("Charlie", "JPY")
        assert new_acc.account_id == "A000003"
        assert restored_bank._acct.n == 4 # Ensure counter continued from 3

        # Test new transaction after restoration
        restored_bank.deposit(acc2.account_id, Decimal("10.00"))
        assert restored_bank.txs[-1].tx_id == "T000006" # Ensure counter continued from 6


    def test_bank_from_dict_empty_or_partial_data(self):
        # Empty dict should create a default bank
        bank = Bank.from_dict({})
        assert bank.name == "MiniBank" # Default name
        assert not bank.accounts
        assert not bank.txs
        assert bank._acct.n == 1
        assert bank._tx.n == 1

        # Partial data
        partial_data = {
            "name": "PartialBank",
            "accounts": {
                "A001": {"account_id": "A001", "owner": "X", "currency": "USD"}
            }
        }
        bank = Bank.from_dict(partial_data)
        assert bank.name == "PartialBank"
        assert "A001" in bank.accounts
        assert bank._acct.n == 1 # Counters not provided, default to 1
        assert bank._tx.n == 1

        # Test with malformed list/dict types in data
        malformed_data = {
            "name": "BadBank",
            "accounts": {"A001": "not a dict"},
            "txs": ["not a dict"],
            "counters": {"acct": "not a dict", "tx": "not a dict"},
        }
        bank = Bank.from_dict(malformed_data)
        assert bank.name == "BadBank"
        assert not bank.accounts # Malformed account skipped
        assert not bank.txs      # Malformed tx skipped
        assert bank._acct.n == 1 # Counters use default
        assert bank._tx.n == 1

        # Test with missing required keys in account dict (will raise TypeError)
        malformed_data_2 = {
            "accounts": {"A001": {"owner": "X", "currency": "USD"}}
        }
        with pytest.raises(TypeError, match="__init__ missing 1 required positional argument: 'account_id'"):
             Bank.from_dict(malformed_data_2)

class TestPersistence:
    def test_load_new_bank(self, tmp_path):
        non_existent_path = tmp_path / "non_existent.json"
        bank = load(str(non_existent_path))
        assert bank.name == "MiniBank"
        assert not bank.accounts
        assert not bank.txs

    def test_save_and_load_existing_bank(self, tmp_path):
        file_path = tmp_path / "bank.json"
        
        # Create a bank, perform operations, save
        bank_orig = Bank("MySavedBank")
        acc1 = bank_orig.open_account("Alice", "USD", initial=Decimal("100"))
        acc2 = bank_orig.open_account("Bob", "EUR")
        bank_orig.deposit(acc1.account_id, Decimal("50"))
        bank_orig.transfer(acc1.account_id, acc2.account_id, Decimal("25"))

        save(bank_orig, str(file_path))

        assert os.path.exists(file_path)
        
        # Load the bank and verify state
        bank_loaded = load(str(file_path))

        assert bank_loaded.name == "MySavedBank"
        assert bank_loaded._acct.n == 3 # A000001, A000002, next is 3
        assert bank_loaded._tx.n == 4 # T000001 (init acc1), T000002 (init acc2), T000003 (deposit), T000004 (transfer)

        loaded_acc1 = bank_loaded.get(acc1.account_id)
        loaded_acc2 = bank_loaded.get(acc2.account_id)

        assert loaded_acc1.bal() == Decimal("125.00") # 100 + 50 - 25
        assert loaded_acc2.bal() == Decimal("25.00")  # 0 + 25

        assert len(bank_loaded.txs) == 4
        assert bank_loaded.txs[0].tx_id == "T000001"
        assert bank_loaded.txs[3].tx_id == "T000004"

        # Ensure atomicity of save (tmp file replaced)
        assert not os.path.exists(str(file_path) + ".tmp")

    def test_load_invalid_json(self, tmp_path):
        file_path = tmp_path / "bad_bank.json"
        with open(file_path, "w") as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            load(str(file_path))

    def test_load_non_dict_json(self, tmp_path):
        file_path = tmp_path / "non_dict_bank.json"
        with open(file_path, "w") as f:
            f.write("[]") # Not a dict at root

        with pytest.raises(Invalid, match="bad storage"):
            load(str(file_path))

class TestDisplayFunctions:
    def test_show_account(self):
        acc = Account("A001", "Alice", "USD", balance="123.45", status="active", created_at="2023-01-01T00:00:00Z")
        expected = "A001 | Alice | USD | 123.45 | active"
        assert show_account(acc) == expected

    def test_show_tx(self):
        tx1 = Tx("T001", "deposit", "100.00", "USD", "2023-01-01T10:00:00Z", dst="A001", note="Salary")
        expected1 = "T001 | 2023-01-01T10:00:00Z | deposit | 100.00 USD | -->A001 | Salary"
        assert show_tx(tx1) == expected1

        tx2 = Tx("T002", "withdraw", "50.00", "EUR", "2023-01-01T10:01:00Z", src="A002", note="")
        expected2 = "T002 | 2023-01-01T10:01:00Z | withdraw | 50.00 EUR | A002-->- |"
        assert show_tx(tx2) == expected2

        tx3 = Tx("T003", "transfer", "25.00", "JPY", "2023-01-01T10:02:00Z", src="A001", dst="A002", note="Gift")
        expected3 = "T003 | 2023-01-01T10:02:00Z | transfer | 25.00 JPY | A001-->A000002 | Gift"
        assert show_tx(tx3) == expected3

class TestReplAndMain:
    # Helper to mock input and capture output
    @pytest.fixture
    def mock_repl_io(self, monkeypatch):
        inputs = []
        outputs = []

        def mock_input(prompt=""):
            outputs.append(prompt) # Capture the prompt
            if not inputs:
                raise EOFError # Simulate Ctrl+D or end of test inputs
            return inputs.pop(0)

        def mock_print(*args, **kwargs):
            outputs.append(" ".join(map(str, args)))

        monkeypatch.setattr('builtins.input', mock_input)
        monkeypatch.setattr('builtins.print', mock_print)
        monkeypatch.setattr('sys.stdout', mock.MagicMock()) # Also mock sys.stdout just in case

        class IOManager:
            def add_input(self, *lines):
                inputs.extend(list(lines))
            
            def get_output(self):
                # Filter out prompts and strip whitespace for easier comparison
                return [line.strip() for line in outputs if not line.endswith("> ")]
            
            def get_full_output(self):
                return outputs

        return IOManager()

    def test_repl_basic_commands(self, tmp_path, mock_repl_io, mock_datetime_now):
        file_path = tmp_path / "minibank.json"
        
        mock_repl_io.add_input(
            "open Alice USD 100",
            "open Bob EUR",
            "accounts",
            "deposit A000001 50 Paycheck",
            "withdraw A000001 25 Rent",
            "transfer A000001 A000002 10 Shared Bill",
            "txs A000001 2", # Test account specific with limit
            "txs 2", # Test general with limit
            "save",
            "quit"
        )
        
        # main() will call repl()
        exit_code = main([ "minibank", str(file_path) ])
        assert exit_code == 0
        
        output = mock_repl_io.get_output()
        
        # Verify initial messages
        assert "MiniBank" in output[0] # From load(path) -> Bank("MiniBank")
        assert "help for commands" in output[1]

        # Verify open commands
        assert f"A000001 | Alice | USD | 100.00 | active" in output[2]
        assert f"A000002 | Bob | EUR | 0.00 | active" in output[3]

        # Verify accounts command
        # Note: balances reflect state *before* subsequent ops shown in output
        assert f"A000001 | Alice | USD | 100.00 | active" in output[4] 
        assert f"A000002 | Bob | EUR | 0.00 | active" in output[5] 
        
        # Verify deposit command
        assert f"T000003 | {FIXED_DATETIME_STR} | deposit | 50.00 USD | -->A000001 | Paycheck" in output[6]
        
        # Verify withdraw command
        assert f"T000004 | {FIXED_DATETIME_STR} | withdraw | 25.00 USD | A000001-->- | Rent" in output[7]

        # Verify transfer command
        assert f"T000005 | {FIXED_DATETIME_STR} | transfer | 10.00 USD | A000001-->A000002 | Shared Bill" in output[8]

        # Verify txs A000001 2 (account specific, limited, reversed)
        # Expected: T000005 (transfer src), T000004 (withdraw)
        assert f"T000005 | {FIXED_DATETIME_STR} | transfer | 10.00 USD | A000001-->A000002 | Shared Bill" in output[9]
        assert f"T000004 | {FIXED_DATETIME_STR} | withdraw | 25.00 USD | A000001-->- | Rent" in output[10]

        # Verify txs 2 (general, limited, reversed)
        # Expected: T000005, T000004
        assert f"T000005 | {FIXED_DATETIME_STR} | transfer | 10.00 USD | A000001-->A000002 | Shared Bill" in output[11]
        assert f"T000004 | {FIXED_DATETIME_STR} | withdraw | 25.00 USD | A000001-->- | Rent" in output[12]

        # Verify save command
        assert "saved" in output[13]

        # Verify that bank state is actually saved
        loaded_bank = load(str(file_path))
        loaded_acc1 = loaded_bank.get("A000001")
        loaded_acc2 = loaded_bank.get("A000002")
        assert loaded_acc1.bal() == Decimal("115.00") # 100 + 50 - 25 - 10
        assert loaded_acc2.bal() == Decimal("10.00")  # 0 + 10
        assert len(loaded_bank.txs) == 5 # 2 init + 3 ops
        assert loaded_bank.txs[-1].tx_id == "T000005"
        
    def test_repl_error_handling(self, tmp_path, mock_repl_io):
        file_path = tmp_path / "minibank.json"
        mock_repl_io.add_input(
            "open Alice USD",
            "deposit A000001 0", # Invalid amount
            "withdraw A000001 1000", # Insufficient funds
            "transfer A000001 A000001 10", # Same account
            "deposit NonExistent 10", # Not found
            "unknown_cmd", # Unknown command
            "quit"
        )
        
        main([ "minibank", str(file_path) ])
        output = mock_repl_io.get_output()

        assert "error: amount must be > 0" in output
        assert "error: insufficient funds" in output
        assert "error: same account" in output
        assert "error: account not found" in output
        assert "error: unknown command" in output

    def test_repl_empty_line_and_eof(self, tmp_path, mock_repl_io):
        file_path = tmp_path / "minibank.json"
        mock_repl_io.add_input(
            "open Alice USD",
            "", # Empty line
            " ", # Whitespace only line
            "quit" # Terminates normally
        )
        main([ "minibank", str(file_path) ])
        output = mock_repl_io.get_output()
        assert "error" not in " ".join(output) # No errors
        assert f"A000001 | Alice | USD | 0.00 | active" in output

        # Test EOF via KeyboardInterrupt simulation
        mock_repl_io.add_input("open Bob EUR") # Add some input so the quit isn't immediate if EOFError is not hit
        # Instead of monkeypatching input to raise EOFError directly, which would circumvent the `try...except` in repl,
        # we simulate it by calling main with an empty input list after setting up the patch, causing the internal `input()` to fail.
        mock_repl_io.add_input() # Ensure the internal list is empty for the second call
        with mock.patch('builtins.input', side_effect=EOFError):
            # Redirect sys.stdout to capture anything printed before quit
            # We're already mocking print, but repl also prints a newline.
            with mock.patch('sys.stdout') as mock_stdout:
                # Need a new path for this scenario, as the previous bank might be saved.
                new_bank_path = tmp_path / "eof_bank.json"
                exit_code = main(["minibank", str(new_bank_path)])
                assert exit_code == 0
                # Verify save was called implicitly by EOFError/KeyboardInterrupt path
                assert os.path.exists(str(new_bank_path))

    def test_main_with_custom_path(self, tmp_path, mock_repl_io):
        custom_path = tmp_path / "custom.json"
        mock_repl_io.add_input("quit")
        main(["minibank", str(custom_path)])
        assert os.path.exists(custom_path) # Should save to custom path
