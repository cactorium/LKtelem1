// a functional library based on Rust iterators!

template <typename It, typename F> struct MapIterator {
  It iterator;
  F &f;

  auto operator*() -> decltype(f(*iterator)) {
    return f(*iterator);
  }
  MapIterator &operator++() {
    iterator++;
    return *this;
  }

  bool operator!=(const MapIterator<It, F>& other) {
    return iterator != other.iterator;
  }
};

template <typename C, typename F> struct MapT {
  C &base;
  F f;

  typedef MapIterator<typename C::const_iterator, F> iterator;
  iterator begin() {
    return iterator{base.begin(), f};
  }
  iterator end() {
    return iterator{base.end(), f};
  }

  template <typename Co> Co collect() {
    Co ret;
    for (auto r : *this) {
      ret.push_back(r);
    }
    return ret;
  }

  template <typename It> It collect(It begin, It end) {
    auto it = begin;
    for (auto r: *this) {
      if (it == end) break;
      *it = r;
    }
    return it;
  }
};

template <typename C, typename F> MapT<C, F> map(C &c, F f) {
    return Map<C, F>{c, f};
}
