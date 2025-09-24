#include"UniquePtr.h"


template<typename T>
UniquePtr<T>::UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

template<typename T>
UniquePtr<T>& UniquePtr<T>::operator=(UniquePtr&& other) noexcept {
    if(this != &other) {
        delete ptr_;
        ptr_ = other.ptr_;
        other.ptr_ == nullptr;
    }
    return *this;
}


template<typename T>
void UniquePtr<T>::reset(T* ptr) {
    delete ptr_;
    ptr_ = ptr;
}

template<typename T>
T* UniquePtr<T>::release() noexcept {
    T* temp = ptr_;
    delete ptr_;
    return temp;
}

template<typename T>
T& UniquePtr<T>::operator*() const {
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

