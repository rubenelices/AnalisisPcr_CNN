import threading
import pytest
from src.campaign_deadline import WallClockDeadline

class FakeEvent:
    def __init__(self,cancel_after_wait=False):
        self.cancelled=False
        self.waits=[]
        self.cancel_after_wait=cancel_after_wait
    def is_set(self):return self.cancelled
    def set(self):self.cancelled=True
    def wait(self,delay):
        self.waits.append(delay)
        if self.cancel_after_wait:self.cancelled=True
        return self.cancelled

def exercise(times,event=None):
    values=iter(times);called=[]
    guard=WallClockDeadline(lambda:next(values),lambda:called.append('stop'))
    guard.stop=event or FakeEvent()
    guard._run()
    return guard,called

def test_wallclock_jump_expires_without_long_relative_wait():
    guard,called=exercise([3600,-40])
    assert guard.stop.waits==[1.0] and called==['stop']

def test_expired_guard_calls_once_without_wait():
    guard,called=exercise([-1])
    assert not guard.stop.waits and called==['stop']

def test_small_remaining_wait_is_bounded():
    guard,called=exercise([.25,0])
    assert guard.stop.waits==[.25] and called==['stop']

def test_cancelled_guard_never_calls():
    event=FakeEvent();event.set()
    guard,called=exercise([],event)
    assert called==[] and guard.stop.waits==[]

def test_cancel_while_waiting_never_calls():
    guard,called=exercise([10],FakeEvent(cancel_after_wait=True))
    assert called==[] and guard.stop.waits==[1]

def test_real_thread_expires_and_can_be_cancelled():
    fired=threading.Event()
    guard=WallClockDeadline(lambda:-1,fired.set)
    guard.start()
    assert fired.wait(timeout=1)
    guard.cancel();guard.thread.join(timeout=1)
    assert not guard.thread.is_alive()

@pytest.mark.parametrize('interval',[0,-1,2,float('nan'),float('inf'),True,'1'])
def test_invalid_interval_rejected(interval):
    with pytest.raises(ValueError):WallClockDeadline(lambda:1,lambda:None,interval)
