# 05 · Latent Factor Model — Algorithm

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §7; reference implementation: [matrix_factorization.py](../../matrix_factorization.py); tài liệu thiết kế: [docs/ML_ENGINE_DESIGN.md](../../docs/ML_ENGINE_DESIGN.md).

## Mô hình

Phân tích ma trận rating `R` thành 2 ma trận latent:

```
R [U × I]  ≈  P [U × K]  ·  Qᵀ [K × I]
```

- `U` = số user (SDT khách), `I` = số món, `K` = số chiều latent (mặc định **10**).
- `R[u][i]` = implicit rating của khách `u` với món `i`.
- `P[u]` = vector latent khách `u`. `Q[i]` = vector latent món `i`.

## Implicit rating từ lịch sử đặt

Không có hệ thống đánh sao. Dùng:

```
R[u][i] = log(1 + số_lần_khách_u_đã_đặt_món_i)
```

| Số lần đặt | R[u][i] |
|---|---|
| 0 | 0.00 |
| 1 | 0.69 |
| 2 | 1.10 |
| 5 | 1.79 |
| 10 | 2.40 |

## Predict

```
R_hat[u][i] = P[u] · Q[i]  =  Σₖ P[u][k] × Q[i][k]
```

## Loss function

```
L = Σ (R[u][i] - P[u]·Q[i])² + λ (‖P‖² + ‖Q‖²)
```

Số hạng đầu đo sai số trên rating đã biết. Số hạng regularization (λ = `REG` = 0.02) chống overfit.

## SGD update rules

Với mỗi triple `(u, i, r_ui)`:

```
e = r_ui - P[u]·Q[i]
P[u] += lr × (e × Q[i]  - reg × P[u])
Q[i] += lr × (e × P[u]  - reg × Q[i])
```

(Khi cập nhật đồng thời, giữ `P[u]` cũ để dùng trong công thức của `Q[i]` — xem `matrix_factorization.py:61-66`.)

## Tham số mặc định

| Tham số | Giá trị | Nơi dùng |
|---|---|---|
| `K` | 10 | Số chiều latent |
| `LR` | 0.01 | Learning rate |
| `REG` | 0.02 | L2 regularization |
| `MAX_ITER` | 50 | Epoch tối đa khi train từ đầu |
| `patience` | 10 | Early stopping: epoch liên tiếp không cải thiện |
| `min_delta` | 1e-4 | Ngưỡng cải thiện loss |

## 3 Luồng cập nhật khi rating thay đổi

Xem `matrix_factorization.py:92-125`, và [docs/ML_ENGINE_DESIGN.md §Tính năng 3](../../docs/ML_ENGINE_DESIGN.md).

| Tỉ lệ ô thay đổi | Chiến lược | Parameters |
|---|---|---|
| **< 10%** | Online update — chỉ SGD trên ô thay đổi | 20 passes, lr bình thường |
| **10-40%** | Fine-tune toàn bộ | `lr × 0.1`, `epoch × 0.2` (min 50) |
| **> 40%** | Train lại từ đầu | Reset P, Q; chạy `fit()` |

## 3 Luồng thêm user / item mới

Xem `matrix_factorization.py:167-269`.

| Tỉ lệ mới | Chiến lược |
|---|---|
| **< 20%** | Chỉ train vector mới, freeze phía còn lại |
| **20-50%** | Train vector mới → fine-tune nhẹ toàn bộ |
| **> 50%** | Reset hoàn toàn, train lại |

## Online update sau mỗi `ORDER_SUBMIT`

Chạy **1 pass SGD** trên các cặp `(userId, itemId)` có trong đơn vừa nhận. Xem `phan-tich-du-an-702.md` §7.6:

```cpp
void onlineUpdate(int userId, int* itemIds, int* quantities, int count) {
    for (int idx = 0; idx < count; idx++) {
        int i = itemIds[idx];
        float r_ui = log(1.0f + orderHistory[userId][i] + quantities[idx]);
        float r_hat = dotProduct(P[userId], Q[i], K);
        float error = r_ui - r_hat;

        for (int k = 0; k < K; k++) {
            float p_old = P[userId][k];
            float q_old = Q[i][k];
            P[userId][k] += LR * (error * q_old - REG * p_old);
            Q[i][k]      += LR * (error * p_old - REG * q_old);
        }

        orderHistory[userId][i] += quantities[idx];
    }
}
```

## Top-K gợi ý

Mỗi khi có `USER_LOGIN` hoặc `ITEM_ADDED`:

1. Tính `score[i] = P[u] · Q[i]` cho mọi món `i`.
2. Loại các món khách đã thêm vào đơn hiện tại (`excluded[]`).
3. Selection sort lấy top-3 (không cần sort toàn bộ — `MAX_MENU ≤ 20`).
4. Gửi `SUGGEST|code1,score1|code2,score2|code3,score3`.

Khách mới (chưa có `P[u]` học được): dùng top món phổ biến nhất làm default — có thể tính bằng `mean(Q, axis=0) · any_direction` hoặc đơn giản là đếm `orderHistory` toàn hệ.

## Mapping Python → C++

| Python (`matrix_factorization.py`) | C++ port (target) | Ghi chú |
|---|---|---|
| `MatrixFactorization(...)` | Global vars `P[]`, `Q[]` + `lfm_init()` | Không wrap class |
| `fit(X)` | `lfm_train()` | Load `R[][]` từ `orderHistory[][]` |
| `_sgd(...)` | `lfm_sgd_pass(...)` | Tham số: `freezeP`, `freezeQ`, `onlyUser`, `onlyItem` |
| `update(X_new)` | `lfm_update_on_change()` | Chạy khi thu ngân sửa rating thủ công (rare) |
| `_online_update(cells)` | `onlineUpdate(userId, items, qty, count)` | **Gọi mỗi `ORDER_SUBMIT`** |
| `add_users(X_ext, n)` | `lfm_add_user(userId)` | Khi `getOrCreateUser()` trả userId mới |
| `add_items(X_ext, n)` | `lfm_add_item(itemIdx)` | Khi `menu.txt` có dòng mới |
| `predict(u, i)` | `predictScore(userId, itemIdx)` | |
| `full_prediction()` | — | Không cần trong server, chỉ để debug |
| `compute_loss(X)` | `lfm_compute_loss()` | In ra khi đóng ca |

## Persistence

| File | Nội dung | Khi nào ghi |
|---|---|---|
| `data/lfm_P.dat` | Binary: `[U][K] × float` | Cuối ca (khi `STOP`) |
| `data/lfm_Q.dat` | Binary: `[I][K] × float` | Cuối ca |
| `data/users.dat` | Mảng SDT + `orderHistory[][]` | Cuối ca |

Load lại khi mở ca tiếp theo (nếu file tồn tại). Nếu không → init ngẫu nhiên nhỏ (`* 0.01`).
