import multiprocessing

class Test:
    count=multiprocessing.Value('i',0)
    
    @classmethod
    def fun(cls):
        cls.count.value+=1

        print('cls1 id : {},cls.count id : {},count:{}'.format(id(cls),id(cls.count),cls.count.value))


    @classmethod
    def test(cls):
        p=multiprocessing.Process(target=cls.fun)
        p.start()
        p.join()
        print('cls2 id : {},cls.count id : {},count:{}'.format(id(cls),id(cls.count),cls.count.value))

if __name__=='__main__':
    Test.test()