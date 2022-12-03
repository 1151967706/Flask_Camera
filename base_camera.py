import  time
import threading

try:
    from greenlet import  getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

class CameraEvent(object):
    #一个类似事件的类，当新帧出现时，它向所有活动客户端发出信号
    def __init__(self):
        self.events={}

    def wait(self):
        ident=get_ident()   #用作回收线程结束后的标识符
        if ident not in self.events:    #如果事件中没有线程，那么创建一个新的线程
            self.events[ident]=[threading.Event(),time.time()]
        return  self.events[ident][0].wait()    #阻塞当前线程

    def set(self):
        now=time.time()
        remove=None
        for ident,event in self.events.items():
            if not event[0].isSet():        #当前事件不存在线程
                event[0].set()      #唤醒线程，将wait的线程唤醒
                event[1]=now
            else:
                if now-event[1]>5:  #如果事件超过5，删除当前线程
                    remove=ident

        if remove:
            del self.events[remove]  #删除变量ident


    def clear(self):
        self.events[get_ident()][0].clear()     #继续阻塞所有的线程

class BaseCamera(object):
    thread=None
    frame=None
    last_access=0
    event= CameraEvent()

    def __init__(self):
        if BaseCamera.thread is None:   #if Camera线程不存在，创建一个新的Camera线程
            BaseCamera.last_access=time.time()
            BaseCamera.thread=threading.Thread(target=self._thread)  #threading.Thread Target用来指定运行的函数
            BaseCamera.thread.start()

            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        BaseCamera.last_access=time.time()

        BaseCamera.event.wait()     #等待Camera线程的信号
        BaseCamera.event.clear()    #阻塞其他线程

        return BaseCamera.frame

    @staticmethod #返回一个静态方法
    def frames():
        #generator 生成器
        raise RuntimeError('Must be implemented by subclasses.')    #手动抛出异常

    @classmethod        #返回一个不许要实例化的类方法
    def _thread(cls):
        print('Starting cmera thread.')     #cls用来表示自身类
        frames_iterator=cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame=frame
            BaseCamera.event.set()
            time.sleep(0)

            if time.time()-BaseCamera.last_access>10:  #
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break

        BaseCamera.thread=None