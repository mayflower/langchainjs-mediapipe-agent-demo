export class AsyncLocalStorage {
  #store;

  getStore() {
    return this.#store;
  }

  run(store, callback) {
    const previous = this.#store;
    this.#store = store;
    try {
      return callback();
    } finally {
      this.#store = previous;
    }
  }

  enterWith(store) {
    this.#store = store;
  }

  disable() {
    this.#store = undefined;
  }
}
