#include <iostream>
#include <string>

using namespace std;

int main() {
    string s1("hello world");
    string s2(s1);
    cout << "before act" << endl;
    cout << "s1 addr " << (const void*)(&s1[0]) << endl;
    cout << "s2 addr " << (const void*)(&s2[0]) << endl;
    s2[0] = 's';
    cout << "after act" << endl;
    cout << "s1 addr " << (const void*)(&s1[0])  << endl;
    cout << "s2 addr " << (const void*)(&s2[0])  << endl;
    cout << "s1 :" << s1 << endl;
    cout << "s2 :" << s2 << endl;
    return 0;
}