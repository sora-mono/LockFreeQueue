#ifndef LOCKFREEQUEUE_H_
#define LOCKFREEQUEUE_H_
#include <atomic>
#include <cassert>
#include <memory>

namespace lockfreequeue {
template <class AsyncObjectType, class T>
class AsyncQueue;
template <class AsyncNotifyType, class T>
class LockFreeQueue;

template <class T>
class AsyncQueue<void, T> {
 protected:
  AsyncQueue(LockFreeQueue<void, T>*) {}
  /// @note linux下gcc和clang不支持constexpr-if，并且不能特化模板类中的模板函数
  /// 通过在基类中声明两个相同的非模板函数来替代
  template <class RespondType>
  inline void RespondAsyncObject(RespondType&& respond) {}
};
/// @brief 通过模板偏特化去除不启用异步通知功能时相关成员变量和函数
/// @note Linux下的gcc和clang过于智障，坚持实例化constexpr-if两分支，所以
/// 只能将RespondAsyncObject拆成两个函数放到基类中
/// @attention 必须保证AsyncQueue的this指针与派生类LockFreeQueue的相同
template <class AsyncObjectType, class T>
class AsyncQueue {
 public:
  virtual ~AsyncQueue() {}
  /// @brief 添加异步通知对象
  /// @param[in] async_notify_object ：异步通知对象
  /// @note 当前仅支持std::promise
  void AddAsyncNotifyObject(AsyncObjectType&& async_notify_object);
  /// @brief 判断异步通知队列是否在狭义上为空
  /// @retval true 队列空
  /// @retval false 队列不空
  /// @note 只计算执行了完整入队流程且未进入出队流程的对象个数
  bool RelaxAsyncNotifyQueueEmpty() const {
    return async_notify_objects_.RelaxEmpty();
  }
  /// @brief 判断队列是否严格为空
  /// @retval true 队列中所有对象已出队且没有对象正在入队
  /// @retval false 队列不空
  bool StrictAsyncNotifyQueueEmpty() const {
    return async_notify_objects_.StrictEmpty();
  }
  /// @brief 获取队列中对象个数
  /// @return 返回队列中执行了完整入队流程且未进入出队流程的对象个数
  size_t AsyncNotifyQueueSize() { return async_notify_objects_.Size(); }

 protected:
  AsyncQueue(LockFreeQueue<AsyncObjectType, T>* full_object_pointer) {
    assert(static_cast<void*>(full_object_pointer) == static_cast<void*>(this));
  }
  /// @brief 响应异步请求
  /// @details 从异步队列中提取一个异步请求并响应
  /// @note Linux下gcc和clang两个智障拒绝识别constexpr-if导致两分支均被实例化，
  /// 通过该函数来避免编译错误，仅用于Enqueue中
  template <class RespondType>
  void RespondAsyncObject(RespondType&& respond) {
    auto async_notify_object = async_notify_objects_.DequeueSpin();
    async_notify_object->set_value(std::forward<RespondType>(respond));
  }
  /// @brief 存储异步通知对象
  LockFreeQueue<void, AsyncObjectType> async_notify_objects_;
};
/// @class LockFreeQueue server.h
/// @brief 无锁化队列
/// @tparam T ：存储的对象类型/对象的基类类型
/// @tparam AsyncObjectType ：异步通知对象类型，void代表不开启异步通知功能
/// @details 进出队列均允许多线程访问
/// 如果启用异步通知功能则异步获取入队对象的优先级高于同步出队的优先级
/// 异步对象获取对象时不严格遵循FIFO规则：异步对象入队时如果成功预留队内对象则
/// 可能比正在等待的异步对象先获得对象
/// 提供给异步对象的结果类型同Dequeue返回值类型
/// @note 开启异步通知会导致入队出队性能下降
/// @attention 析构过程必须保证没有线程正在执行出队/入队操作
template <class AsyncObjectType, class T>
class LockFreeQueue : public AsyncQueue<AsyncObjectType, T> {
 public:
  LockFreeQueue() : AsyncQueue<AsyncObjectType, T>(this) {
    Wrapper* base_node = new Wrapper{nullptr, nullptr};
    head_.store(base_node, std::memory_order_release);
    object_count_.store(0, std::memory_order_release);
    tail_.store(base_node, std::memory_order_release);
  }
  ~LockFreeQueue();
  /// @brief 入队一个对象
  /// @param[in] args :构造对象的参数
  template <class ObjectType = T, class... Args>
  void Enqueue(Args&&... args);
  /// @brief 出队一个对象
  /// @return 返回指向对象的指针
  /// @retval nullptr ：当前无对象执行了完整的入队流程
  [[nodiscard]] std::unique_ptr<T> Dequeue();
  /// @brief 出队一个对象，如果对象不存在则持续自旋
  /// @return 返回指向对象的指针
  /// @retval nullptr ：（永远不应该返回该值）
  [[nodiscard]] std::unique_ptr<T> DequeueSpin();
  /// @brief 获取队列中对象个数
  /// @return 返回队列中执行了完整入队流程且未进入出队流程的对象个数
  /// @note 使用acquire语义
  int64_t Size() const { return object_count_.load(std::memory_order_acquire); }
  /// @brief 判断队列是否在狭义上为空
  /// @retval true 队列空
  /// @retval false 队列不空
  /// @note 只计算执行了完整入队流程且未进入出队流程的对象个数
  /// 使用acquire语义
  bool RelaxEmpty() const {
    return object_count_.load(std::memory_order_acquire) == 0;
  }
  /// @brief 判断队列是否严格为空
  /// @retval true 队列中所有对象已出队且没有对象正在入队
  /// @retval false 队列不空
  bool StrictEmpty() const {
    return RelaxEmpty() && head_.load(std::memory_order_acquire) ==
                               tail_.load(std::memory_order_acquire);
  }

 private:
  friend AsyncQueue<AsyncObjectType, T>;

  struct Wrapper {
    std::atomic<T*> object;
    std::atomic<Wrapper*> next;
  };

  /// @brief 出队对象但不削减对象计数
  /// @return 返回指向对象的指针
  /// @retval nullptr ：（永远不应该返回该值）
  /// @details 自旋至有对象出队
  /// @attention 需要保证一定有对象存在于队列中或将执行完整入队过程
  std::unique_ptr<T> DequeueNoDeduceObjectCount();
  /// @brief 已完全添加的对象
  std::atomic<int64_t> object_count_;
  /// @brief 队列头，指向队列中上次出队的节点
  /// @details 通过使用哨兵机制防止head_指向nullptr，否则可能出现获取了出队资格
  /// 但head_指向nullptr这种情况
  std::atomic<Wrapper*> head_;
  /// @brief 队列尾，next指向最近插入的节点
  /// @note 语义：永远指向最近插入的节点，无论该节点是否完成完整入队/出队过程
  std::atomic<Wrapper*> tail_;
};

template <class AsyncObjectType, class T>
inline void lockfreequeue::AsyncQueue<AsyncObjectType, T>::AddAsyncNotifyObject(
    AsyncObjectType&& async_notify_object) {
  static_assert(!std::is_same_v<AsyncObjectType, void>);
  auto full_object_pointer =
      static_cast<LockFreeQueue<AsyncObjectType, T>*>(this);
  // 预留对象计数
  int64_t old_object_count =
      full_object_pointer->object_count_.load(std::memory_order_acquire);
  while (!full_object_pointer->object_count_.compare_exchange_weak(
      old_object_count, old_object_count - 1, std::memory_order_release,
      std::memory_order_acquire)) {
  }
  if (old_object_count > 0) {
    // 已经预留队列中完整入队的对象，可以直接出队
    async_notify_object.set_value(full_object_pointer->DequeueSpin());
  } else {
    // 队伍中无完整入队的对象，等待Enqueue函数执行完毕后异步检查
    async_notify_objects_.Enqueue(
        std::forward<AsyncObjectType>(async_notify_object));
  }
}
template <class AsyncObjectType, class T>
template <class ObjectType, class... Args>
inline void LockFreeQueue<AsyncObjectType, T>::Enqueue(Args&&... args) {
  Wrapper* wrapper = new Wrapper{
      .object = new ObjectType(std::forward<Args>(args)...), .next = nullptr};
  // 尝试修改尾指针为构造好的新节点
  Wrapper* last_node = tail_.load(std::memory_order_acquire);
  while (!tail_.compare_exchange_weak(last_node, wrapper,
                                      std::memory_order_release,
                                      std::memory_order_acquire)) {
  }
  // 修改尾指针后更新它的前一个节点的next指针
  last_node->next.store(wrapper, std::memory_order_release);
  // 增加有效节点计数
  if constexpr (std::is_same_v<AsyncObjectType, void>) {
    // 未启用异步通知功能
    object_count_.fetch_add(1, std::memory_order_acq_rel);
  } else {
    // 启用异步通知功能
    int64_t old_object_count = object_count_.load(std::memory_order_acquire);
    while (!object_count_.compare_exchange_weak(
        old_object_count, old_object_count + 1, std::memory_order_release,
        std::memory_order_acquire)) {
    }
    if (old_object_count < 0) {
      // 有正在等待新节点入队的异步通知对象
      // 出队一个对象和一个异步通知对象，将对象交给异步对象
      AsyncQueue<AsyncObjectType, T>::RespondAsyncObject(
          DequeueNoDeduceObjectCount());
    }
  }
}
template <class AsyncObjectType, class T>
inline lockfreequeue::LockFreeQueue<AsyncObjectType, T>::~LockFreeQueue() {
  // 以下均为单线程操作且无其它线程入队/出队
  Wrapper* head = head_.load(std::memory_order_relaxed);
  Wrapper* tail = tail_.load(std::memory_order_relaxed);
  while (head != tail) {
    Wrapper* temp = head;
    head = head->next;
    delete temp;
    delete head->object;
  }
  delete head;
}
template <class AsyncObjectType, class T>
inline std::unique_ptr<T> LockFreeQueue<AsyncObjectType, T>::Dequeue() {
  // 获取计数（获取出队资格）
  int64_t last_object_count = object_count_.load(std::memory_order_acquire);
  // 减少计数
  while (last_object_count > 0 &&
         !object_count_.compare_exchange_weak(
             last_object_count, last_object_count - 1,
             std::memory_order_release, std::memory_order_acquire)) {
  }
  if (last_object_count <= 0) {
    return nullptr;
  }
  return DequeueNoDeduceObjectCount();
}
template <class AsyncObjectType, class T>
inline std::unique_ptr<T> LockFreeQueue<AsyncObjectType, T>::DequeueSpin() {
  // 获取计数（获取出队资格）
  int64_t last_object_count = object_count_.load(std::memory_order_acquire);
  // 减少计数
  while (!object_count_.compare_exchange_weak(
      last_object_count, last_object_count - 1, std::memory_order_release,
      std::memory_order_acquire)) {
  }
  return DequeueNoDeduceObjectCount();
}
template <class AsyncObjectType, class T>
inline std::unique_ptr<T>
LockFreeQueue<AsyncObjectType, T>::DequeueNoDeduceObjectCount() {
  // 让头指针指向当前出队节点
  Wrapper* last_invalid_node = head_.load(std::memory_order_acquire);
  Wrapper* last_invalid_node_next;
  // last_invalid_node->next为nullptr时需要自旋，等待新节点被连上
  do {
    last_invalid_node_next =
        last_invalid_node->next.load(std::memory_order_acquire);
  } while (last_invalid_node_next == nullptr ||
           !head_.compare_exchange_weak(
               last_invalid_node, last_invalid_node_next,
               std::memory_order_release, std::memory_order_acquire));

  T* result = last_invalid_node_next->object.exchange(
      nullptr, std::memory_order_acq_rel);
  // 只有这个节点的对象指针被获取后才可以安全删除
  // 防止另一个线程获取了last_invalid_node后还未提取object指针时就释放
  // TODO ：使用内存管理器可以优化释放过程.实现使用了循环队列的部分原理，
  // 出队仅移动头指针，所有无效节点都自然连成一个单链表，所以可以通过从单链表头
  // 重用节点的方式避免释放
  while (last_invalid_node->object.load(std::memory_order_acquire) != nullptr) {
  }
  delete last_invalid_node;
  assert(result != nullptr);
  return std::unique_ptr<T>(result);
}

}  // namespace lockfreequeue

#endif  // !SERVER_LOCKFREEQUEUE_H_
