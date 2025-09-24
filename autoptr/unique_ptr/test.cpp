#include <iostream>
#include "UniquePtr.h" // 假设UniquePtr类定义在UniquePtr.h中
 
struct MyClass {
    MyClass(int value) : value_(value) {}
    ~MyClass() { std::cout << "Destroying MyClass with value " << value_ << std::endl; }
    int value_;
};
 
int main() {
    UniquePtr<MyClass> ptr1(new MyClass(10)); // 使用UniquePtr管理MyClass对象
    std::cout << "ptr1 points to " << ptr1->value_ << std::endl; // 使用箭头操作符访问成员变量
    (*ptr1).value_ = 20; // 使用解引用操作符修改成员变量值，这里仅为示例，通常更推荐使用箭头操作符来访问成员变量和成员函数。
    std::cout << "ptr1 points to " << ptr1->value_ << std::endl; // 输出修改后的值以验证修改成功。
    return 0; // 当main函数返回时，ptr1将被销毁，并自动删除所指向的MyClass对象。
} // 在此处输出“Destroying MyClass with value 20”以验证对象已被正确删除。