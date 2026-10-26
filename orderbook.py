from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(slots=True)
class Order:
    id: int
    side: Side
    price: int
    qty: int

    @property
    def live(self) -> bool:
        return self.qty > 0


@dataclass(frozen=True, slots=True)
class Fill:
    maker_id: int
    taker_id: int
    price: int
    qty: int


class _BookSide:
    def __init__(self, side: Side) -> None:
        self.side = side
        self._levels: dict[int, deque[Order]] = {}
        self._heap: list[int] = []
        self._sign = -1 if side is Side.BUY else 1

    def add(self, order: Order) -> None:
        level = self._levels.get(order.price)
        if level is None:
            level = self._levels[order.price] = deque()
            heapq.heappush(self._heap, self._sign * order.price)
        level.append(order)

    def best_price(self) -> int | None:
        while self._heap:
            price = self._sign * self._heap[0]
            level = self._levels[price]
            while level and not level[0].live:
                level.popleft()
            if level:
                return price
            heapq.heappop(self._heap)
            del self._levels[price]
        return None

    def front(self, price: int) -> Order:
        return self._levels[price][0]

    def crosses(self, price: int, limit: int | None) -> bool:
        if limit is None:
            return True
        return price <= limit if self.side is Side.SELL else price >= limit

    def depth(self, n: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for price in sorted(self._levels, key=lambda p: self._sign * p):
            total = sum(o.qty for o in self._levels[price] if o.live)
            if total:
                out.append((price, total))
                if len(out) == n:
                    break
        return out


class OrderBook:
    def __init__(self) -> None:
        self._sides = {Side.BUY: _BookSide(Side.BUY), Side.SELL: _BookSide(Side.SELL)}
        self._resting: dict[int, Order] = {}
        self._next_id = 1

    def submit(self, side: Side, price: int, qty: int) -> tuple[int, list[Fill]]:
        if price <= 0 or qty <= 0:
            raise ValueError("price and qty must be positive")
        order = self._new_order(side, price, qty)
        fills = list(self._match(order, price))
        if order.live:
            self._sides[side].add(order)
            self._resting[order.id] = order
        return order.id, fills

    def submit_market(self, side: Side, qty: int) -> tuple[int, list[Fill]]:
        if qty <= 0:
            raise ValueError("qty must be positive")
        order = self._new_order(side, 0, qty)
        return order.id, list(self._match(order, None))

    def cancel(self, order_id: int) -> bool:
        order = self._resting.pop(order_id, None)
        if order is None:
            return False
        order.qty = 0
        return True

    def amend(self, order_id: int, new_qty: int) -> bool:
        order = self._resting.get(order_id)
        if order is None or not 0 < new_qty <= order.qty:
            return False
        order.qty = new_qty
        return True

    def best_bid(self) -> int | None:
        return self._sides[Side.BUY].best_price()

    def best_ask(self) -> int | None:
        return self._sides[Side.SELL].best_price()

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else ask - bid

    def depth(self, side: Side, levels: int = 5) -> list[tuple[int, int]]:
        return self._sides[side].depth(levels)

    def _new_order(self, side: Side, price: int, qty: int) -> Order:
        order = Order(self._next_id, side, price, qty)
        self._next_id += 1
        return order

    def _match(self, taker: Order, limit: int | None) -> Iterator[Fill]:
        book = self._sides[taker.side.opposite]
        while taker.live:
            price = book.best_price()
            if price is None or not book.crosses(price, limit):
                return
            maker = book.front(price)
            qty = min(taker.qty, maker.qty)
            taker.qty -= qty
            maker.qty -= qty
            if not maker.live:
                del self._resting[maker.id]
            yield Fill(maker.id, taker.id, price, qty)


def test_price_time_priority() -> None:
    book = OrderBook()
    a, _ = book.submit(Side.SELL, 101, 5)
    b, _ = book.submit(Side.SELL, 101, 3)
    book.submit(Side.SELL, 102, 4)
    _, fills = book.submit(Side.BUY, 101, 6)
    assert [(f.maker_id, f.qty) for f in fills] == [(a, 5), (b, 1)]
    assert book.depth(Side.SELL) == [(101, 2), (102, 4)]


def test_sweep_and_rest() -> None:
    book = OrderBook()
    book.submit(Side.SELL, 101, 2)
    book.submit(Side.SELL, 102, 4)
    _, fills = book.submit(Side.BUY, 105, 10)
    assert [(f.price, f.qty) for f in fills] == [(101, 2), (102, 4)]
    assert book.depth(Side.BUY) == [(105, 4)]
    assert book.best_ask() is None and book.best_bid() == 105


def test_cancel_middle_of_queue() -> None:
    book = OrderBook()
    a, _ = book.submit(Side.SELL, 100, 1)
    b, _ = book.submit(Side.SELL, 100, 1)
    c, _ = book.submit(Side.SELL, 100, 1)
    assert book.cancel(b)
    assert not book.cancel(b)
    _, fills = book.submit_market(Side.BUY, 2)
    assert [f.maker_id for f in fills] == [a, c]


def test_market_order_never_rests() -> None:
    book = OrderBook()
    book.submit(Side.SELL, 100, 2)
    book.submit(Side.SELL, 101, 3)
    _, fills = book.submit_market(Side.BUY, 4)
    assert [(f.price, f.qty) for f in fills] == [(100, 2), (101, 2)]
    _, fills = book.submit_market(Side.BUY, 10)
    assert [(f.price, f.qty) for f in fills] == [(101, 1)]
    assert book.best_ask() is None and book.best_bid() is None


def test_amend_keeps_priority() -> None:
    book = OrderBook()
    a, _ = book.submit(Side.SELL, 100, 5)
    b, _ = book.submit(Side.SELL, 100, 5)
    assert book.amend(a, 2)
    assert not book.amend(b, 10)
    assert not book.amend(b, 0)
    _, fills = book.submit_market(Side.BUY, 3)
    assert [(f.maker_id, f.qty) for f in fills] == [(a, 2), (b, 1)]
    assert not book.amend(a, 1)


def test_spread() -> None:
    book = OrderBook()
    book.submit(Side.BUY, 99, 1)
    book.submit(Side.SELL, 102, 1)
    assert book.spread() == 3
    book.submit(Side.BUY, 102, 1)
    assert book.spread() is None


if __name__ == "__main__":
    for test in [
        test_price_time_priority,
        test_sweep_and_rest,
        test_cancel_middle_of_queue,
        test_market_order_never_rests,
        test_amend_keeps_priority,
        test_spread,
    ]:
        test()
    print("all tests passed")