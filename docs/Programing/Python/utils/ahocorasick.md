https://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html

```python title="ahocorasick.py"
# coding=utf-8
import collections
import queue


class State(object):
    def __init__(self, depth=0):
        self.depth = depth
        self.root_state = self if depth == 0 else None
        self.success = collections.OrderedDict()
        self.failure = None
        self.emits = set()

    def add_emit(self, keyword):
        self.emits.add(keyword)

    def add_emits(self, emits):
        for emit in emits:
            self.add_emit(emit)

    def set_failure(self, fail_state):
        self.failure = fail_state

    def get_next_state(self, token, ignore_root_state=False):
        next_state = self.success.get(token)
        if not ignore_root_state and next_state is None and self.root_state is not None:
            next_state = self.root_state
        return next_state

    def add_state(self, token):
        next_state = self.get_next_state(token, ignore_root_state=True)
        if next_state is None:
            next_state = State(self.depth + 1)
            self.success[token] = next_state
        return next_state

    def get_states(self):
        return self.success.values()

    def get_transitions(self):
        return self.success.keys()


class Trie(object):
    def __init__(self, allow_overlaps=True, case_insensitive=False):
        self.root_state = State()
        self.failure_states_constructed = False
        self.allow_overlaps = allow_overlaps
        self.case_insensitive = case_insensitive

    def build_trie(self, keyword_list):
        for keyword in keyword_list:
            self._add_key_word(keyword)
        self._constructed_failure_states()

    def parse_text(self, text):
        current_state = self.root_state
        collected_intervals = []
        for i, token in enumerate(text):
            if self.case_insensitive:
                token = token.lower()
            current_state = self._get_state(current_state, token)
            self._store_intervals(i, current_state, collected_intervals)

        if not self.allow_overlaps:
            interval_tree = IntervalTree(collected_intervals)
            interval_tree.remove_overlaps(collected_intervals)
        return collected_intervals

    def has_keyword(self, text):
        current_state = self.root_state
        for token in text:
            if self.case_insensitive:
                token = token.lower()
            next_state = self._get_state(current_state, token)
            if next_state is not None and next_state != current_state and len(next_state.emits) != 0:
                return True
            current_state = next_state
        return False

    def _add_key_word(self, keyword):
        if keyword is None or len(keyword) == 0:
            return
        current_state = self.root_state
        for token in keyword:
            if self.case_insensitive:
                token = token.lower()
            current_state = current_state.add_state(token)
        current_state.add_emit(keyword)

    def _get_state(self, current_state, token):
        # 先按照success跳转
        new_current_state = current_state.get_next_state(token, ignore_root_state=False)
        # 跳转失败的话，按failure跳转
        while new_current_state is None:
            current_state = current_state.failure
            new_current_state = current_state.get_next_state(token, ignore_root_state=False)
        return new_current_state

    def _constructed_failure_states(self):
        queue_state = queue.Queue()

        # 第一步，将深度为1的节点的failure设为根节点
        for depth_one_state in self.root_state.get_states():
            depth_one_state.set_failure(self.root_state)
            queue_state.put(depth_one_state)

        # 第二步，为深度>1的节点建立failure表（通过bfs实现）
        while not queue_state.empty():
            current_state = queue_state.get()

            for transition in current_state.get_transitions():
                target_state = current_state.get_next_state(transition, ignore_root_state=False)
                queue_state.put(target_state)

                trace_failure_state = current_state.failure
                while trace_failure_state.get_next_state(transition, ignore_root_state=False) is None:
                    trace_failure_state = trace_failure_state.failure
                new_failure_state = trace_failure_state.get_next_state(transition, ignore_root_state=False)
                target_state.set_failure(new_failure_state)
                target_state.add_emits(new_failure_state.emits)
        self.failure_states_constructed = True

    def _store_intervals(self, position, current_state, collected_intervals):
        emits = current_state.emits
        for emit in emits:
            collected_intervals.append(Interval(position - len(emit) + 1, position, emit))


class IntervalTree(object):
    def __init__(self, intervals):
        self.root_node = IntervalNode(intervals)

    def remove_overlaps(self, intervals):
        intervals.sort(key=lambda interval: (-interval.size(), interval.start))

        remove_intervals = set()
        for interval in intervals:
            if interval in remove_intervals:
                continue
            overlaps = self.find_overlaps(interval)
            for overlap in overlaps:
                remove_intervals.add(overlap)

        for remove_interval in remove_intervals:
            intervals.remove(remove_interval)

        intervals.sort(key=lambda interval: interval.start)
        return intervals

    def find_overlaps(self, interval):
        return self.root_node.find_overlaps(interval)


class IntervalNode(object):
    def __init__(self, intervals):
        self.left = None
        self.right = None
        self.intervals = []

        self.point = self.determine_median(intervals)

        # 以中点为界靠左的区间
        to_left = []
        # 以中点为界靠右的区间
        to_right = []

        for interval in intervals:
            if interval.end < self.point:
                to_left.append(interval)
            elif interval.start > self.point:
                to_right.append(interval)
            else:
                self.intervals.append(interval)

        if len(to_left) > 0:
            self.left = IntervalNode(to_left)
        if len(to_right) > 0:
            self.right = IntervalNode(to_right)

    def determine_median(self, intervals):
        start = -1
        end = -1
        for interval in intervals:
            current_start = interval.start
            current_end = interval.end
            if start == -1 or current_start < start:
                start = current_start
            if end == -1 or current_end > end:
                end = current_end
        return (start + end) / 2

    def find_overlaps(self, interval):
        overlaps = []

        if self.point < interval.start:
            self.add_to_overlaps(interval, overlaps, self.find_overlapping_ranges(self.right, interval))
            self.add_to_overlaps(interval, overlaps, self.check_for_overlaps_to_the_right(interval))
        elif self.point > interval.end:
            self.add_to_overlaps(interval, overlaps, self.find_overlapping_ranges(self.left, interval))
            self.add_to_overlaps(interval, overlaps, self.check_for_overlaps_to_the_left(interval))
        else:
            self.add_to_overlaps(interval, overlaps, self.intervals)
            self.add_to_overlaps(interval, overlaps, self.find_overlapping_ranges(self.left, interval))
            self.add_to_overlaps(interval, overlaps, self.find_overlapping_ranges(self.right, interval))
        return overlaps

    def add_to_overlaps(self, interval, overlaps, new_overlaps):
        for current_interval in new_overlaps:
            if current_interval != interval:
                overlaps.append(current_interval)

    def check_for_overlaps_to_the_left(self, interval):
        overlaps = []
        for current_interval in self.intervals:
            if current_interval.start <= interval.end:
                overlaps.append(current_interval)
        return overlaps

    def check_for_overlaps_to_the_right(self, interval):
        overlaps = []
        for current_interval in self.intervals:
            if current_interval.end >= interval.start:
                overlaps.append(current_interval)
        return overlaps

    def find_overlapping_ranges(self, node, interval):
        if node is not None:
            return node.find_overlaps(interval)
        return []


class Interval(object):
    def __init__(self, start, end, keyword=""):
        self.start = start
        self.end = end
        self.keyword = keyword

    def size(self):
        return self.end - self.start + 1

    def overlaps_with_interval(self, other):
        return self.start <= other.end and self.end >= other.start

    def overlaps_with_point(self, point):
        return self.start <= point <= self.end

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return self.start % 100 + self.end % 100

    def __cmp__(self, other):
        if not isinstance(other, Interval):
            return -1
        comparison = self.start - other.start
        return comparison if comparison != 0 else (self.end - other.end)

    def __str__(self):
        return str(self.start) + ":" + str(self.end) + "=" + self.keyword
```