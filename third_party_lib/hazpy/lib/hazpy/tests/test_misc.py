from hazpy.misc import FancyList, batch, chop_at, columnize, unique

def test_main():
    # Test chop_at().
    s = "123\nabc"
    lis = FancyList([(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D')])
    assert lis.column(1) == ['A', 'B', 'C', 'D']
    assert chop_at(s, "3\n") == "12"
    assert chop_at(s, "3\n", True) == "123\n"

    # Test columnize().
    arg = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    expected = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
    action = columnize(arg, 3)
    assert action == expected, action
    arg.append(16)
    expected = [[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16]]
    action = columnize(arg, 3)
    assert action == expected, action
    arg.append(17)
    expected = [[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17]]
    action = columnize(arg, 3)
    assert action == expected, action
    arg.append(18)
    expected = [[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18]]
    action = columnize(arg, 3)
    assert action == expected, action
    arg.append(19)
    expected = [[1,2,3,4,5,6,7], [8,9,10,11,12,13,14], [15,16,17,18,19]]
    action = columnize(arg, 3)
    assert action == expected, action

def test_batch_1():
    control = ["ABCDE", "FGHIJ", "KLM"]
    iterable = "abcdefghijklm"
    def action(b):
        return "".join(b).upper()
    result = batch(iterable, 5, action)
    assert result == control

def test_batch_2():
    control = [[ord("A"), ord("B")], [ord("C"), ord("D")]]
    iterable = "abcd"
    def prepare_element(elm):
        return elm.upper()
    def action(b):
        return [ord(x) for x in b]
    result = batch(iterable, 2, action, prepare_element)
    assert result == control
    
def test_unique():
    assert unique([1, 4, 1, 2]) == [1, 4, 2]
