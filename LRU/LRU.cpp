/*
采用解耦的架构 哈希表 + 双向链表 为什么使用这两种数据结构？
涉及到快速查找使用哈希表记录并查找 哈希表存的应该是一个对应地址
通过查key 获得指针并进一步返回数据
为什么使用双向链表？方便队头队尾插入删除
是为了满足LRU性质 对数据排列 最久未使用的可以清出cache

链表->双向链表(便于插入删除 修改操作)
    + hashmap -> LRU cache;
*/
#include <iostream>
#include <unordered_map>
using namespace std;


struct ListNode {
    ListNode* prev;
    ListNode* next;
    int key;
    int value;

    ListNode(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
};

//实现查找元素 清楚元素 清空链表
class DoublyLinkedList {
private:
    ListNode* head;
    ListNode* tail;
public:
    //构造函数
    DoublyLinkedList() {
        head = new ListNode(0, 0);
        tail = new ListNode(0, 0);
        //是双向链表 别写成循环链表了
        head->next = tail;
        tail->prev = head;
    }

    //析构函数
    ~DoublyLinkedList() {
        //释放所有的节点 满足RAII规范
        clear();
        delete head;
        delete tail;
    }

    //移动最近使用数据到队头
    void move_to_front(ListNode* node) {
        node->next->prev = node->prev;
        node->prev->next = node->next;
        push_front(node);
    }

    //入链表到表头
    void push_front(ListNode* node) {
        head->next->prev = node;
        node->next = head->next;
        node->prev = head;
        head->next = node;
    }

    //删除某个元素
    void erase(ListNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
    }

    //弹出队尾元素
    ListNode* pop_back() {
        ListNode* node = tail->prev;
        if(node != head) {
            tail->prev = node->prev;
            node->prev->next = tail;
            node->next = nullptr;
            node->prev = nullptr;
            return node;
        }
        return nullptr;
    }
    //清空队列
    void clear() {
        ListNode* node = head->next;
        while(node != tail) {
            ListNode* next = node->next;
            erase(node);
            node = next;
        }
    }

};

//实现get put
class LRUCache {
private:
    DoublyLinkedList _list;
    int _capacity;

    std::unordered_map<int,ListNode*> _cache;

public:
    LRUCache(int capacity) : _capacity(capacity) {}

    int get(int key) {
        auto it = _cache.find(key);
        if(it != _cache.end()) {
            _list.move_to_front(it->second);
            return it->second->value;
        } else {
            return -1;
        }
    }

    void put(int k, int v) {
        //先找这块缓存在不在
        auto it = _cache.find(k);
        if(it != _cache.end()) {
            it->second->value = v;
            _list.move_to_front(it->second);
        } else {
            if(_cache.size() == _capacity) {    
                //容量超了 需要弹出最久未使用 为了复用设计pop_back 只弹出不删除
                ListNode* node = _list.pop_back();
                if(node != nullptr) {
                    _cache.erase(node->key);
                }
            }
            ListNode* newnode = new ListNode(k, v);
            _cache[k] = newnode;
            _list.push_front(newnode);
        }
    }
};


void test()
{
    LRUCache cache(2);
 
    cache.put(1, 1); // 缓存现在为 {1=1}
    cache.put(2, 2); // 缓存现在为 {1=1, 2=2}
    std::cout << "Get 1: " << cache.get(1) << std::endl; // 返回 1
    cache.put(3, 3); // 缓存达到容量，移除最近最少使用的键 2，缓存现在为 {1=1, 3=3}
    std::cout << "Get 2: " << cache.get(2) << std::endl; // 返回 -1（未找到）
    cache.put(4, 4); // 缓存达到容量，移除最近最少使用的键 1，缓存现在为 {3=3, 4=4}
    std::cout << "Get 1: " << cache.get(1) << std::endl; // 返回 -1（未找到）
    std::cout << "Get 3: " << cache.get(3) << std::endl; // 返回 3
    std::cout << "Get 4: " << cache.get(4) << std::endl; // 返回 4
}

int main() {
    test();
    return 0;
}
