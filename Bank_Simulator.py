from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional


def now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def money(v: str | int | float | Decimal) -> Decimal:
    if isinstance(v, Decimal):
        x = v
    else:
        x = Decimal(str(v))
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class BankError(Exception):
    pass


class NotFound(BankError):
    pass


class Invalid(BankError):
    pass


class InsufficientFunds(BankError):
    pass


@dataclass
class Account:
    account_id: str
    owner: str
    currency: str
    balance: str = "0.00"
    status: str = "active"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = now()

    def bal(self) -> Decimal:
        return Decimal(self.balance)

    def set_bal(self, v: Decimal) -> None:
        self.balance = str(money(v))


@dataclass(frozen=True)
class Tx:
    tx_id: str
    kind: str
    amount: str
    currency: str
    ts: str
    src: Optional[str] = None
    dst: Optional[str] = None
    note: str = ""


class Counter:
    def __init__(self, prefix: str, n: int = 1) -> None:
        self.prefix = prefix
        self.n = n

    def next(self) -> str:
        v = f"{self.prefix}{self.n:06d}"
        self.n += 1
        return v

    def snap(self) -> Dict[str, str | int]:
        return {"prefix": self.prefix, "n": self.n}

    @staticmethod
    def from_snap(d: Dict[str, str | int]) -> "Counter":
        return Counter(str(d["prefix"]), int(d["n"]))


class Bank:
    def __init__(self, name: str) -> None:
        self.name = name
        self.accounts: Dict[str, Account] = {}
        self.txs: List[Tx] = []
        self._acct = Counter("A")
        self._tx = Counter("T")

    def open_account(self, owner: str, currency: str, initial: Decimal = Decimal("0")) -> Account:
        owner = owner.strip()
        currency = currency.strip().upper()
        if not owner:
            raise Invalid("owner required")
        if len(currency) != 3:
            raise Invalid("currency invalid")
        aid = self._acct.next()
        a = Account(account_id=aid, owner=owner, currency=currency)
        self.accounts[aid] = a
        if money(initial) > Decimal("0"):
            self.deposit(aid, money(initial), note="initial")
        return a

    def get(self, account_id: str) -> Account:
        account_id = account_id.strip()
        if account_id not in self.accounts:
            raise NotFound("account not found")
        return self.accounts[account_id]

    def _active(self, account_id: str) -> Account:
        a = self.get(account_id)
        if a.status != "active":
            raise Invalid("account not active")
        return a

    def deposit(self, account_id: str, amount: Decimal, note: str = "") -> Tx:
        a = self._active(account_id)
        amt = money(amount)
        if amt <= Decimal("0"):
            raise Invalid("amount must be > 0")
        a.set_bal(a.bal() + amt)
        tx = Tx(
            tx_id=self._tx.next(),
            kind="deposit",
            amount=str(amt),
            currency=a.currency,
            ts=now(),
            dst=a.account_id,
            note=note.strip(),
        )
        self.txs.append(tx)
        return tx

    def withdraw(self, account_id: str, amount: Decimal, note: str = "") -> Tx:
        a = self._active(account_id)
        amt = money(amount)
        if amt <= Decimal("0"):
            raise Invalid("amount must be > 0")
        if a.bal() < amt:
            raise InsufficientFunds("insufficient funds")
        a.set_bal(a.bal() - amt)
        tx = Tx(
            tx_id=self._tx.next(),
            kind="withdraw",
            amount=str(amt),
            currency=a.currency,
            ts=now(),
            src=a.account_id,
            note=note.strip(),
        )
        self.txs.append(tx)
        return tx

    def transfer(self, src: str, dst: str, amount: Decimal, note: str = "") -> Tx:
        if src.strip() == dst.strip():
            raise Invalid("same account")
        a = self._active(src)
        b = self._active(dst)
        if a.currency != b.currency:
            raise Invalid("currency mismatch")
        amt = money(amount)
        if amt <= Decimal("0"):
            raise Invalid("amount must be > 0")
        if a.bal() < amt:
            raise InsufficientFunds("insufficient funds")
        a.set_bal(a.bal() - amt)
        b.set_bal(b.bal() + amt)
        tx = Tx(
            tx_id=self._tx.next(),
            kind="transfer",
            amount=str(amt),
            currency=a.currency,
            ts=now(),
            src=a.account_id,
            dst=b.account_id,
            note=note.strip(),
        )
        self.txs.append(tx)
        return tx

    def list_accounts(self) -> List[Account]:
        return list(self.accounts.values())

    def list_txs(self, account_id: Optional[str] = None, limit: int = 30) -> List[Tx]:
        if limit <= 0:
            raise Invalid("limit must be > 0")
        if account_id is None:
            return list(reversed(self.txs))[:limit]
        self.get(account_id)
        out: List[Tx] = []
        for tx in reversed(self.txs):
            if tx.src == account_id or tx.dst == account_id:
                out.append(tx)
                if len(out) >= limit:
                    break
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "accounts": {k: asdict(v) for k, v in self.accounts.items()},
            "txs": [asdict(t) for t in self.txs],
            "counters": {"acct": self._acct.snap(), "tx": self._tx.snap()},
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Bank":
        b = Bank(str(d.get("name", "Bank")))
        counters = d.get("counters", {})
        if isinstance(counters, dict):
            if "acct" in counters and isinstance(counters["acct"], dict):
                b._acct = Counter.from_snap(counters["acct"])
            if "tx" in counters and isinstance(counters["tx"], dict):
                b._tx = Counter.from_snap(counters["tx"])
        accounts = d.get("accounts", {})
        if isinstance(accounts, dict):
            for k, v in accounts.items():
                if isinstance(v, dict):
                    b.accounts[k] = Account(**v)
        txs = d.get("txs", [])
        if isinstance(txs, list):
            for v in txs:
                if isinstance(v, dict):
                    b.txs.append(Tx(**v))
        return b


def load(path: str) -> Bank:
    if not os.path.exists(path):
        return Bank("MiniBank")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise Invalid("bad storage")
    return Bank.from_dict(d)


def save(bank: Bank, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bank.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def show_account(a: Account) -> str:
    return f"{a.account_id} | {a.owner} | {a.currency} | {a.balance} | {a.status}"


def show_tx(tx: Tx) -> str:
    s = tx.src or "-"
    d = tx.dst or "-"
    return f"{tx.tx_id} | {tx.ts} | {tx.kind} | {tx.amount} {tx.currency} | {s}->{d} | {tx.note}"


def parse_amount(s: str) -> Decimal:
    s = s.strip()
    if not s:
        raise Invalid("amount required")
    return money(Decimal(s))


def repl(path: str) -> int:
    bank = load(path)
    print(bank.name)
    print("help for commands")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            save(bank, path)
            return 0

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in {"quit", "exit"}:
                save(bank, path)
                return 0

            if cmd == "help":
                print("open <owner> <currency> [initial]")
                print("accounts")
                print("deposit <account_id> <amount> [note...]")
                print("withdraw <account_id> <amount> [note...]")
                print("transfer <src> <dst> <amount> [note...]")
                print("txs [account_id] [limit]")
                print("save")
                print("quit")
                continue

            if cmd == "save":
                save(bank, path)
                print("saved")
                continue

            if cmd == "open":
                if len(args) < 2:
                    raise Invalid("usage: open <owner> <currency> [initial]")
                owner = args[0]
                ccy = args[1]
                initial = parse_amount(args[2]) if len(args) >= 3 else Decimal("0")
                a = bank.open_account(owner, ccy, initial=initial)
                print(show_account(a))
                continue

            if cmd == "accounts":
                for a in bank.list_accounts():
                    print(show_account(a))
                continue

            if cmd == "deposit":
                if len(args) < 2:
                    raise Invalid("usage: deposit <account_id> <amount> [note...]")
                aid = args[0]
                amt = parse_amount(args[1])
                note = " ".join(args[2:])
                tx = bank.deposit(aid, amt, note=note)
                print(show_tx(tx))
                continue

            if cmd == "withdraw":
                if len(args) < 2:
                    raise Invalid("usage: withdraw <account_id> <amount> [note...]")
                aid = args[0]
                amt = parse_amount(args[1])
                note = " ".join(args[2:])
                tx = bank.withdraw(aid, amt, note=note)
                print(show_tx(tx))
                continue

            if cmd == "transfer":
                if len(args) < 3:
                    raise Invalid("usage: transfer <src> <dst> <amount> [note...]")
                src = args[0]
                dst = args[1]
                amt = parse_amount(args[2])
                note = " ".join(args[3:])
                tx = bank.transfer(src, dst, amt, note=note)
                print(show_tx(tx))
                continue

            if cmd == "txs":
                account_id = args[0] if len(args) >= 1 else None
                limit = int(args[1]) if len(args) >= 2 else 30
                for tx in bank.list_txs(account_id=account_id, limit=limit):
                    print(show_tx(tx))
                continue

            raise Invalid("unknown command")

        except BankError as e:
            print(f"error: {e}")
        except Exception as e:
            print(f"fatal: {type(e).__name__}: {e}")

    return 0


def main(argv: List[str]) -> int:
    path = "minibank.json"
    if len(argv) >= 2:
        path = argv[1]
    return repl(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
