#ifndef PandaTree_Framework_Iterator_h
#define PandaTree_Framework_Iterator_h

#include <type_traits>

namespace panda {
  namespace utils {

    //! Iterator class for containers.
    template<class C, bool is_const>
    class Iterator {
      typedef C container_type;
      typedef Iterator<container_type, false> mutable_type;
      typedef Iterator<container_type, true> const_type;

      friend container_type;
      friend const_type;

    public:
      typedef Iterator<C, is_const> self_type;
      typedef typename std::conditional<is_const, typename C::const_reference, typename C::reference>::type reference;
      typedef typename std::conditional<is_const, typename C::const_pointer, typename C::pointer>::type pointer;

      Iterator() {}
      Iterator(mutable_type const& it) : ptr_(it.ptr_), unitSize_(it.unitSize_) {}
      self_type& operator++() { shiftPtr_(1); return *this; }
      self_type operator++(int) { auto itr(*this); shiftPtr_(1); return itr; }
      self_type& operator--() { shiftPtr_(-1); return *this; }
      self_type operator--(int) { auto itr(*this); shiftPtr_(-1); return itr; }
      self_type& operator+=(int n) { shiftPtr_(n); return *this; }
      self_type& operator-=(int n) { shiftPtr_(-n); return *this; }
      self_type operator+(int n) const { auto itr(*this); return (itr += n); }
      self_type operator-(int n) const { auto itr(*this); return (itr -= n); }
      bool operator==(mutable_type const& rhs) const { return ptr_ == rhs.ptr_; }
      bool operator==(const_type const& rhs) const { return ptr_ == rhs.ptr_; }
      bool operator!=(mutable_type const& rhs) const { return ptr_ != rhs.ptr_; }
      bool operator!=(const_type const& rhs) const { return ptr_ != rhs.ptr_; }
      reference operator*() const { return *ptr_; }
      pointer operator->() const { return ptr_; }

    private:
      Iterator(pointer p, unsigned unitSize) : ptr_(p), unitSize_(unitSize) {}

      typedef typename std::conditional<is_const, char const*, char*>::type char_pointer;

      void shiftPtr_(int shift) { reinterpret_cast<char_pointer&>(ptr_) += unitSize_ * shift; }

      pointer ptr_{0};
      unsigned const unitSize_{0};
    };

  }
}

#endif
