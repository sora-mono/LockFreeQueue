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
  /// @note linux��gcc��clang��֧��constexpr-if�����Ҳ����ػ�ģ�����е�ģ�庯��
  /// ͨ���ڻ���������������ͬ�ķ�ģ�庯�������
  template <class RespondType>
  inline void RespondAsyncObject(RespondType&& respond) {}
};
/// @brief ͨ��ģ��ƫ�ػ�ȥ���������첽֪ͨ����ʱ��س�Ա�����ͺ���
/// @note Linux�µ�gcc��clang�������ϣ����ʵ����constexpr-if����֧������
/// ֻ�ܽ�RespondAsyncObject������������ŵ�������
/// @attention ���뱣֤AsyncQueue��thisָ����������LockFreeQueue����ͬ
template <class AsyncObjectType, class T>
class AsyncQueue {
 public:
  virtual ~AsyncQueue() {}
  /// @brief ����첽֪ͨ����
  /// @param[in] async_notify_object ���첽֪ͨ����
  /// @note ��ǰ��֧��std::promise
  void AddAsyncNotifyObject(AsyncObjectType&& async_notify_object);
  /// @brief �ж��첽֪ͨ�����Ƿ���������Ϊ��
  /// @retval true ���п�
  /// @retval false ���в���
  /// @note ֻ����ִ�����������������δ����������̵Ķ������
  bool RelaxAsyncNotifyQueueEmpty() const {
    return async_notify_objects_.RelaxEmpty();
  }
  /// @brief �ж϶����Ƿ��ϸ�Ϊ��
  /// @retval true ���������ж����ѳ�����û�ж����������
  /// @retval false ���в���
  bool StrictAsyncNotifyQueueEmpty() const {
    return async_notify_objects_.StrictEmpty();
  }
  /// @brief ��ȡ�����ж������
  /// @return ���ض�����ִ�����������������δ����������̵Ķ������
  size_t AsyncNotifyQueueSize() { return async_notify_objects_.Size(); }

 protected:
  AsyncQueue(LockFreeQueue<AsyncObjectType, T>* full_object_pointer) {
    assert(static_cast<void*>(full_object_pointer) == static_cast<void*>(this));
  }
  /// @brief ��Ӧ�첽����
  /// @details ���첽��������ȡһ���첽������Ӧ
  /// @note Linux��gcc��clang�������Ͼܾ�ʶ��constexpr-if��������֧����ʵ������
  /// ͨ���ú��������������󣬽�����Enqueue��
  template <class RespondType>
  void RespondAsyncObject(RespondType&& respond) {
    auto async_notify_object = async_notify_objects_.DequeueSpin();
    async_notify_object->set_value(std::forward<RespondType>(respond));
  }
  /// @brief �洢�첽֪ͨ����
  LockFreeQueue<void, AsyncObjectType> async_notify_objects_;
};
/// @class LockFreeQueue server.h
/// @brief ����������
/// @tparam T ���洢�Ķ�������/����Ļ�������
/// @tparam AsyncObjectType ���첽֪ͨ�������ͣ�void���������첽֪ͨ����
/// @details �������о�������̷߳���
/// ��������첽֪ͨ�������첽��ȡ��Ӷ�������ȼ�����ͬ�����ӵ����ȼ�
/// �첽�����ȡ����ʱ���ϸ���ѭFIFO�����첽�������ʱ����ɹ�Ԥ�����ڶ�����
/// ���ܱ����ڵȴ����첽�����Ȼ�ö���
/// �ṩ���첽����Ľ������ͬDequeue����ֵ����
/// @note �����첽֪ͨ�ᵼ����ӳ��������½�
/// @attention �������̱��뱣֤û���߳�����ִ�г���/��Ӳ���
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
  /// @brief ���һ������
  /// @param[in] args :�������Ĳ���
  template <class ObjectType = T, class... Args>
  void Enqueue(Args&&... args);
  /// @brief ����һ������
  /// @return ����ָ������ָ��
  /// @retval nullptr ����ǰ�޶���ִ�����������������
  [[nodiscard]] std::unique_ptr<T> Dequeue();
  /// @brief ����һ������������󲻴������������
  /// @return ����ָ������ָ��
  /// @retval nullptr ������Զ��Ӧ�÷��ظ�ֵ��
  [[nodiscard]] std::unique_ptr<T> DequeueSpin();
  /// @brief ��ȡ�����ж������
  /// @return ���ض�����ִ�����������������δ����������̵Ķ������
  /// @note ʹ��acquire����
  int64_t Size() const { return object_count_.load(std::memory_order_acquire); }
  /// @brief �ж϶����Ƿ���������Ϊ��
  /// @retval true ���п�
  /// @retval false ���в���
  /// @note ֻ����ִ�����������������δ����������̵Ķ������
  /// ʹ��acquire����
  bool RelaxEmpty() const {
    return object_count_.load(std::memory_order_acquire) == 0;
  }
  /// @brief �ж϶����Ƿ��ϸ�Ϊ��
  /// @retval true ���������ж����ѳ�����û�ж����������
  /// @retval false ���в���
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

  /// @brief ���Ӷ��󵫲������������
  /// @return ����ָ������ָ��
  /// @retval nullptr ������Զ��Ӧ�÷��ظ�ֵ��
  /// @details �������ж������
  /// @attention ��Ҫ��֤һ���ж�������ڶ����л�ִ��������ӹ���
  std::unique_ptr<T> DequeueNoDeduceObjectCount();
  /// @brief ����ȫ��ӵĶ���
  std::atomic<int64_t> object_count_;
  /// @brief ����ͷ��ָ��������ϴγ��ӵĽڵ�
  /// @details ͨ��ʹ���ڱ����Ʒ�ֹhead_ָ��nullptr��������ܳ��ֻ�ȡ�˳����ʸ�
  /// ��head_ָ��nullptr�������
  std::atomic<Wrapper*> head_;
  /// @brief ����β��nextָ���������Ľڵ�
  /// @note ���壺��Զָ���������Ľڵ㣬���۸ýڵ��Ƿ�����������/���ӹ���
  std::atomic<Wrapper*> tail_;
};

template <class AsyncObjectType, class T>
inline void lockfreequeue::AsyncQueue<AsyncObjectType, T>::AddAsyncNotifyObject(
    AsyncObjectType&& async_notify_object) {
  static_assert(!std::is_same_v<AsyncObjectType, void>);
  auto full_object_pointer =
      static_cast<LockFreeQueue<AsyncObjectType, T>*>(this);
  // Ԥ���������
  int64_t old_object_count =
      full_object_pointer->object_count_.load(std::memory_order_acquire);
  while (!full_object_pointer->object_count_.compare_exchange_weak(
      old_object_count, old_object_count - 1, std::memory_order_release,
      std::memory_order_acquire)) {
  }
  if (old_object_count > 0) {
    // �Ѿ�Ԥ��������������ӵĶ��󣬿���ֱ�ӳ���
    async_notify_object.set_value(full_object_pointer->DequeueSpin());
  } else {
    // ��������������ӵĶ��󣬵ȴ�Enqueue����ִ����Ϻ��첽���
    async_notify_objects_.Enqueue(
        std::forward<AsyncObjectType>(async_notify_object));
  }
}
template <class AsyncObjectType, class T>
template <class ObjectType, class... Args>
inline void LockFreeQueue<AsyncObjectType, T>::Enqueue(Args&&... args) {
  Wrapper* wrapper = new Wrapper{
      .object = new ObjectType(std::forward<Args>(args)...), .next = nullptr};
  // �����޸�βָ��Ϊ����õ��½ڵ�
  Wrapper* last_node = tail_.load(std::memory_order_acquire);
  while (!tail_.compare_exchange_weak(last_node, wrapper,
                                      std::memory_order_release,
                                      std::memory_order_acquire)) {
  }
  // �޸�βָ����������ǰһ���ڵ��nextָ��
  last_node->next.store(wrapper, std::memory_order_release);
  // ������Ч�ڵ����
  if constexpr (std::is_same_v<AsyncObjectType, void>) {
    // δ�����첽֪ͨ����
    object_count_.fetch_add(1, std::memory_order_acq_rel);
  } else {
    // �����첽֪ͨ����
    int64_t old_object_count = object_count_.load(std::memory_order_acquire);
    while (!object_count_.compare_exchange_weak(
        old_object_count, old_object_count + 1, std::memory_order_release,
        std::memory_order_acquire)) {
    }
    if (old_object_count < 0) {
      // �����ڵȴ��½ڵ���ӵ��첽֪ͨ����
      // ����һ�������һ���첽֪ͨ���󣬽����󽻸��첽����
      AsyncQueue<AsyncObjectType, T>::RespondAsyncObject(
          DequeueNoDeduceObjectCount());
    }
  }
}
template <class AsyncObjectType, class T>
inline lockfreequeue::LockFreeQueue<AsyncObjectType, T>::~LockFreeQueue() {
  // ���¾�Ϊ���̲߳������������߳����/����
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
  // ��ȡ��������ȡ�����ʸ�
  int64_t last_object_count = object_count_.load(std::memory_order_acquire);
  // ���ټ���
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
  // ��ȡ��������ȡ�����ʸ�
  int64_t last_object_count = object_count_.load(std::memory_order_acquire);
  // ���ټ���
  while (!object_count_.compare_exchange_weak(
      last_object_count, last_object_count - 1, std::memory_order_release,
      std::memory_order_acquire)) {
  }
  return DequeueNoDeduceObjectCount();
}
template <class AsyncObjectType, class T>
inline std::unique_ptr<T>
LockFreeQueue<AsyncObjectType, T>::DequeueNoDeduceObjectCount() {
  // ��ͷָ��ָ��ǰ���ӽڵ�
  Wrapper* last_invalid_node = head_.load(std::memory_order_acquire);
  Wrapper* last_invalid_node_next;
  // last_invalid_node->nextΪnullptrʱ��Ҫ�������ȴ��½ڵ㱻����
  do {
    last_invalid_node_next =
        last_invalid_node->next.load(std::memory_order_acquire);
  } while (last_invalid_node_next == nullptr ||
           !head_.compare_exchange_weak(
               last_invalid_node, last_invalid_node_next,
               std::memory_order_release, std::memory_order_acquire));

  T* result = last_invalid_node_next->object.exchange(
      nullptr, std::memory_order_acq_rel);
  // ֻ������ڵ�Ķ���ָ�뱻��ȡ��ſ��԰�ȫɾ��
  // ��ֹ��һ���̻߳�ȡ��last_invalid_node��δ��ȡobjectָ��ʱ���ͷ�
  // TODO ��ʹ���ڴ�����������Ż��ͷŹ���.ʵ��ʹ����ѭ�����еĲ���ԭ��
  // ���ӽ��ƶ�ͷָ�룬������Ч�ڵ㶼��Ȼ����һ�����������Կ���ͨ���ӵ�����ͷ
  // ���ýڵ�ķ�ʽ�����ͷ�
  while (last_invalid_node->object.load(std::memory_order_acquire) != nullptr) {
  }
  delete last_invalid_node;
  assert(result != nullptr);
  return std::unique_ptr<T>(result);
}

}  // namespace lockfreequeue

#endif  // !SERVER_LOCKFREEQUEUE_H_
