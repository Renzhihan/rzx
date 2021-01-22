
import multiprocessing

def print_1(num):
    print("cube: {}".format(num*num))

def print_2(num):
    print("square: {}".format(num*num*num))


if __name__=="__main__":
    p1=multiprocessing.Process(target=print_1,args=(10,))
    p2=multiprocessing.Process(target=print_2,args=(10,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("done")