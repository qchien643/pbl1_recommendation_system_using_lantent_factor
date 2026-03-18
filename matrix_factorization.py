import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors=2, learning_rate=0.01, reg=0.02, n_epochs=5000,
                 random_state=42, patience=10, min_delta=1e-4):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.patience  = patience    # số epoch chịu đựng không cải thiện
        self.min_delta = min_delta   # cải thiện nhỏ hơn ngưỡng này → coi như không cải thiện

        self.P = None
        self.Q = None
        self.loss_history = []
        self.X_snapshot = None      # lưu lại bảng rating gốc để so sánh sau này

    # =========================================================================
    # CORE
    # =========================================================================

    def fit(self, X, missing_value=0):
        np.random.seed(self.random_state)
        n_users, n_items = X.shape

        self.P = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        self.X_snapshot = X.copy()

        self._sgd(X, missing_value, self.learning_rate, self.n_epochs, verbose_every=500)

    def _sgd(self, X, missing_value, lr, n_epochs, verbose_every=500,
             freeze_P=False, freeze_Q=False,
             only_users=None, only_items=None):
        """
        Lõi SGD dùng chung cho mọi trường hợp.
        freeze_P / freeze_Q : không cập nhật ma trận đó
        only_users / only_items : chỉ tính rating liên quan đến tập chỉ định
        """
        known_ratings = []
        n_users, n_items = X.shape
        for i in range(n_users):
            for j in range(n_items):
                if X[i, j] != missing_value:
                    if only_users is not None and i not in only_users:
                        continue
                    if only_items is not None and j not in only_items:
                        continue
                    known_ratings.append((i, j, X[i, j]))

        best_loss        = np.inf
        no_improve_count = 0

        for epoch in range(n_epochs):
            np.random.shuffle(known_ratings)
            for i, j, x_ij in known_ratings:
                x_hat   = np.dot(self.P[i], self.Q[j])
                e_ij    = x_ij - x_hat
                p_i_old = self.P[i].copy()

                if not freeze_P:
                    self.P[i] += lr * (e_ij * self.Q[j] - self.reg * self.P[i])
                if not freeze_Q:
                    self.Q[j] += lr * (e_ij * p_i_old   - self.reg * self.Q[j])

            loss = self.compute_loss(X, missing_value)
            self.loss_history.append(loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}, Loss = {loss:.6f}")

            # --- Early stopping ---
            if loss < best_loss - self.min_delta:
                best_loss        = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print(f"  [Early Stopping] Dừng tại epoch {epoch + 1}/{n_epochs} "
                      f"— Loss không cải thiện hơn {self.min_delta} "
                      f"trong {self.patience} epoch liên tiếp")
                print(f"  [Early Stopping] Loss tốt nhất: {best_loss:.6f}")
                break

    # =========================================================================
    # TÍNH NĂNG 1: Cập nhật khi người dùng chỉnh sửa dữ liệu
    # =========================================================================

    def update(self, X_new, missing_value=0):
        """
        Gọi khi bảng rating bị chỉnh sửa.
        Tự động phân luồng dựa trên tỉ lệ ô thay đổi / tổng ô đã biết.

        Phân luồng:
          < 10%  → online update: SGD chỉ trên các ô thay đổi
          10–40% → fine-tune toàn bộ với lr và epoch nhỏ hơn
          > 40%  → train lại từ đầu
        """
        if self.X_snapshot is None:
            raise RuntimeError("Chưa fit model. Hãy gọi fit() trước.")

        changed_cells = self._find_changed_cells(self.X_snapshot, X_new, missing_value)
        total_known   = int(np.sum(X_new != missing_value))
        change_ratio  = len(changed_cells) / total_known if total_known > 0 else 0

        print(f"\n[Update] Số ô thay đổi : {len(changed_cells)}/{total_known} "
              f"({change_ratio*100:.1f}%)")

        if change_ratio < 0.10:
            print("[Update] Luồng 1 — Online update (thay đổi nhỏ < 10%)")
            self._online_update(changed_cells)

        elif change_ratio < 0.40:
            print("[Update] Luồng 2 — Fine-tune toàn bộ (thay đổi trung bình 10–40%)")
            self._finetune(X_new, missing_value, lr_scale=0.1, epoch_scale=0.2)

        else:
            print("[Update] Luồng 3 — Train lại từ đầu (thay đổi lớn > 40%)")
            self.fit(X_new, missing_value)
            return  # fit() đã cập nhật snapshot

        self.X_snapshot = X_new.copy()

    def _find_changed_cells(self, X_old, X_new, missing_value):
        """Trả về list (i, j, giá_trị_mới) của các ô bị thay đổi."""
        changed = []
        for i in range(X_new.shape[0]):
            for j in range(X_new.shape[1]):
                old_val = X_old[i, j]
                new_val = X_new[i, j]
                if old_val != new_val and new_val != missing_value:
                    changed.append((i, j, new_val))
        return changed

    def _online_update(self, changed_cells, n_passes=20):
        """
        Luồng 1: Chỉ chạy SGD đúng trên các ô thay đổi.
        n_passes: số lần lặp qua các ô thay đổi.
        """
        for _ in range(n_passes):
            for i, j, x_ij in changed_cells:
                x_hat   = np.dot(self.P[i], self.Q[j])
                e_ij    = x_ij - x_hat
                p_i_old = self.P[i].copy()
                self.P[i] += self.learning_rate * (e_ij * self.Q[j] - self.reg * self.P[i])
                self.Q[j] += self.learning_rate * (e_ij * p_i_old   - self.reg * self.Q[j])

        print(f"  Đã update {len(changed_cells)} ô, {n_passes} lần lặp.")

    def _finetune(self, X, missing_value, lr_scale=0.1, epoch_scale=0.2):
        """
        Luồng 2: Fine-tune toàn bộ P, Q với lr và epoch nhỏ hơn.
        lr_scale / epoch_scale: tỉ lệ so với giá trị gốc.
        """
        lr     = self.learning_rate * lr_scale
        epochs = max(50, int(self.n_epochs * epoch_scale))
        print(f"  Fine-tune {epochs} epochs, lr={lr:.5f}")
        self._sgd(X, missing_value, lr=lr, n_epochs=epochs, verbose_every=0)

    # =========================================================================
    # TÍNH NĂNG 2: Thêm user hoặc item mới
    # =========================================================================

    def add_users(self, X_extended, n_new_users, missing_value=0):
        """
        Thêm n_new_users hàng mới vào cuối bảng.
        X_extended: bảng rating mới kích thước (n_users + n_new_users, n_items)

        Phân luồng theo tỉ lệ user mới / tổng user:
          < 20%  → chỉ train vector mới, Q cố định
          20–50% → train vector mới, rồi fine-tune toàn bộ
          > 50%  → train lại từ đầu
        """
        n_old_users = self.P.shape[0]
        ratio = n_new_users / (n_old_users + n_new_users)

        print(f"\n[Add Users] Thêm {n_new_users} users mới "
              f"(tổng cũ: {n_old_users}, tỉ lệ: {ratio*100:.1f}%)")

        # Khởi tạo vector latent cho user mới và ghép vào P
        P_new = np.random.normal(scale=0.1, size=(n_new_users, self.n_factors))
        self.P = np.vstack([self.P, P_new])
        new_user_indices = set(range(n_old_users, n_old_users + n_new_users))

        if ratio < 0.20:
            print("[Add Users] Luồng 1 — Chỉ train vector user mới, Q cố định")
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate,
                      n_epochs=max(200, self.n_epochs // 5),
                      verbose_every=0,
                      freeze_Q=True,
                      only_users=new_user_indices)

        elif ratio < 0.50:
            print("[Add Users] Luồng 2 — Train vector mới rồi fine-tune toàn bộ")
            # Bước 1: train riêng user mới, Q không đổi
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate,
                      n_epochs=max(200, self.n_epochs // 5),
                      verbose_every=0,
                      freeze_Q=True,
                      only_users=new_user_indices)
            # Bước 2: fine-tune nhẹ toàn bộ để nhất quán
            self._finetune(X_extended, missing_value, lr_scale=0.1, epoch_scale=0.1)

        else:
            print("[Add Users] Luồng 3 — Train lại từ đầu (quá nhiều user mới)")
            n_users, n_items = X_extended.shape
            self.P = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
            self.Q = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate, n_epochs=self.n_epochs, verbose_every=500)

        self.X_snapshot = X_extended.copy()
        print(f"  Hoàn tất. P shape: {self.P.shape}")

    def add_items(self, X_extended, n_new_items, missing_value=0):
        """
        Thêm n_new_items cột mới vào cuối bảng.
        X_extended: bảng rating mới kích thước (n_users, n_items + n_new_items)

        Phân luồng theo tỉ lệ item mới / tổng item:
          < 20%  → chỉ train vector mới, P cố định
          20–50% → train vector mới, rồi fine-tune toàn bộ
          > 50%  → train lại từ đầu
        """
        n_old_items = self.Q.shape[0]
        ratio = n_new_items / (n_old_items + n_new_items)

        print(f"\n[Add Items] Thêm {n_new_items} items mới "
              f"(tổng cũ: {n_old_items}, tỉ lệ: {ratio*100:.1f}%)")

        # Khởi tạo vector latent cho item mới và ghép vào Q
        Q_new = np.random.normal(scale=0.1, size=(n_new_items, self.n_factors))
        self.Q = np.vstack([self.Q, Q_new])
        new_item_indices = set(range(n_old_items, n_old_items + n_new_items))

        if ratio < 0.20:
            print("[Add Items] Luồng 1 — Chỉ train vector item mới, P cố định")
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate,
                      n_epochs=max(200, self.n_epochs // 5),
                      verbose_every=0,
                      freeze_P=True,
                      only_items=new_item_indices)

        elif ratio < 0.50:
            print("[Add Items] Luồng 2 — Train vector mới rồi fine-tune toàn bộ")
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate,
                      n_epochs=max(200, self.n_epochs // 5),
                      verbose_every=0,
                      freeze_P=True,
                      only_items=new_item_indices)
            self._finetune(X_extended, missing_value, lr_scale=0.1, epoch_scale=0.1)

        else:
            print("[Add Items] Luồng 3 — Train lại từ đầu (quá nhiều item mới)")
            n_users, n_items = X_extended.shape
            self.P = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
            self.Q = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
            self._sgd(X_extended, missing_value,
                      lr=self.learning_rate, n_epochs=self.n_epochs, verbose_every=500)

        self.X_snapshot = X_extended.copy()
        print(f"  Hoàn tất. Q shape: {self.Q.shape}")

    # =========================================================================
    # PREDICT & LOSS
    # =========================================================================

    def predict(self, i, j):
        return np.dot(self.P[i], self.Q[j])

    def full_prediction(self, clip_min=0, clip_max=5):
        Y = np.dot(self.P, self.Q.T)
        return np.clip(Y, clip_min, clip_max)

    def compute_loss(self, X, missing_value=0):
        loss = 0.0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] != missing_value:
                    loss += (X[i, j] - np.dot(self.P[i], self.Q[j])) ** 2
        loss += self.reg * (np.sum(self.P ** 2) + np.sum(self.Q ** 2))
        return loss


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    X = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [0, 0, 5, 4],
    ], dtype=float)

    model = MatrixFactorization(n_factors=2, learning_rate=0.01,
                                reg=0.02, n_epochs=5000, random_state=42,
                                patience=10, min_delta=1e-4)

    print("=" * 60)
    print("BƯỚC 1: Train ban đầu")
    print("=" * 60)
    model.fit(X)
    print(f"Loss sau train: {model.compute_loss(X):.4f}")

    # ------------------------------------------------------------------
    # Demo update: 1 ô thay đổi → Luồng 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 2: Chỉnh sửa 1 ô (< 10%) → Luồng 1")
    print("=" * 60)
    X_v2 = X.copy()
    X_v2[0, 1] = 4                  # user 0 đổi rating item 1: 3 → 4
    model.update(X_v2)

    # ------------------------------------------------------------------
    # Demo update: 3 ô thay đổi → Luồng 2
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 3: Chỉnh sửa 3 ô (~30%) → Luồng 2")
    print("=" * 60)
    X_v3 = X_v2.copy()
    X_v3[1, 0] = 2
    X_v3[2, 3] = 3
    X_v3[3, 3] = 2
    model.update(X_v3)

    # ------------------------------------------------------------------
    # Demo thêm 1 user mới (~20%) → Luồng 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 4: Thêm 1 user mới (16.7%) → Luồng 1")
    print("=" * 60)
    X_v4 = np.vstack([model.X_snapshot, [3, 0, 4, 0]])
    model.add_users(X_v4, n_new_users=1)

    # ------------------------------------------------------------------
    # Demo thêm 1 item mới (~16.7%) → Luồng 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 5: Thêm 1 item mới (16.7%) → Luồng 1")
    print("=" * 60)
    new_col = np.array([[0], [5], [0], [3], [0]])
    X_v5 = np.hstack([model.X_snapshot, new_col])
    model.add_items(X_v5, n_new_items=1)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    print(f"P shape: {model.P.shape}, Q shape: {model.Q.shape}")
    print("\nMa trận dự đoán:")
    print(np.round(model.full_prediction(), 3))
