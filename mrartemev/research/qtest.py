from multiprocessing import Process, Queue
import time

def f(q):
    time.sleep(3)
    q.put([42, None, 'hello'])
    print('mem', q.get())
def r(q):
    q.put(2)
    q.put(3)

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    r(q)
    print(q.get())    # prints "[42, None, 'hello']"
    #p.join()