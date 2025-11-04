#pragma once


template<typename T>
class UniquePtr {
public:

    //构造函数
    //使用显式声明禁止隐式转换
    explicit UniquePtr(T* ptr = nullptr) : ptr_(ptr) {}

    //禁用拷贝构造函数和拷贝赋值构造函数
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(UniquePtr& other) = delete;

    //允许移动构造函数和移动赋值构造函数
    UniquePtr(UniquePtr&& other) noexcept;
    UniquePtr& operator=(UniquePtr&& other) noexcept;

    //重置指针(用别的指针)
    void reset(T* ptr = nullptr);

    //获取原始指针
    T* get() const {
        return ptr_;
    }

    //释放指针
    T* release() noexcept;

    //解引用操作符
    T& operator*() const;
    T* operator->() const;

    //指针判空
    explicit operator bool() const {
        return ptr_ != nullptr;
    }

    //析构函数
    ~UniquePtr();

private:
    T* ptr_;

};



template<typename T>
UniquePtr<T>::UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

template<typename T>
UniquePtr<T>& UniquePtr<T>::operator=(UniquePtr&& other) noexcept {
    if(this != &other) {
        delete ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
    }
    return *this;
}


template<typename T>
void UniquePtr<T>::reset(T* ptr) {
    delete ptr_;
    ptr_ = ptr;
}

//release只是释放所有权，把管理权转移出去，不销毁对象
//比如有些接口不适合用智能指针 所以用release把内存转移出去
template<typename T>
T* UniquePtr<T>::release() noexcept {
    T* temp = ptr_;
    // 只是释放所有权，不删除对象，标识unique_ptr不在管理这快内存
    ptr_ = nullptr;  // 只是释放所有权，不删除对象，标识unique_ptr不在管理这快内存
    return temp;
}

template<typename T>
T& UniquePtr<T>::operator*() const {
    //解引用操作 直接把内存保管的值送回去
    return *ptr_;
}

template<typename T>
T* UniquePtr<T>::operator->() const {
    return ptr_;
}

template<typename T>
UniquePtr<T>::~UniquePtr() {
    delete ptr_;
}

