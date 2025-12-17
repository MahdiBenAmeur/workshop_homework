# MiniBank: A Simple In-Memory Bank Simulator

## Project Overview

This project implements **MiniBank**, a straightforward, in-memory bank simulation system designed to showcase core banking functionalities. It allows users to perform essential operations such as opening accounts, managing deposits, withdrawals, and transfers, and maintaining a detailed transaction history.

The state of the bank (all accounts and transactions) can be persistently saved to and loaded from a JSON file, ensuring data integrity across sessions.

An interactive command-line interface (REPL) provides a user-friendly way to interact with the bank simulator, enabling real-time management of bank operations.

### Key Features

- **Monetary Precision:** All financial calculations use Python’s `Decimal` type to prevent floating-point inaccuracies, with values rounded to two decimal places using `ROUND_HALF_UP`.
- **Immutable Transactions:** Transaction records (`Tx` objects) are immutable once created, ensuring an auditable history.
- **Unique Identifiers:** Accounts and transactions receive unique sequential IDs (e.g. `A000001`, `T000001`).
- **Atomic Persistence:** Bank state is saved atomically using a temporary file to minimize corruption risks.
- **Comprehensive Validation:** Validations include positive amounts, sufficient funds, active account status, and currency matching.
- **User-Friendly REPL:** Clear commands, immediate feedback, and graceful error handling via custom exceptions.
- **UTC Timestamping:** All timestamps use UTC in ISO 8601 format.

## Project Structure

```
.
├── test_files/
│   └── test_bank_simulator.py
├── Bank_Simulator.py
├── README.md
└── minibank.json
```

## Setup / Installation

### Prerequisites
- Python 3.8 or newer

### Installation
```bash
pip install pytest
```

## Usage
```bash
python Bank_Simulator.py
```

## Running Tests
```bash
pytest test_files/
```
