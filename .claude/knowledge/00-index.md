# Knowledge Base Index

Bản đồ tra cứu: chủ đề → file KB → mục trong [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md).

## Theo chủ đề nghiệp vụ

| Chủ đề | File KB | Mục gốc |
|---|---|---|
| Dự án là gì, có ai tham gia | [01-overview.md](01-overview.md) | §1, §4.1 |
| 16 business rules | [02-business-rules.md](02-business-rules.md) | §4.2 |
| Bảng mã món ăn | [03-menu-codes.md](03-menu-codes.md) | §2.2 |
| Nguyên tắc UX không gõ tiếng Việt | [07-ux-cli-design.md](07-ux-cli-design.md) | §2 |
| Mockup giao diện CLI | [07-ux-cli-design.md](07-ux-cli-design.md) | §15 |
| Mở/đóng ca, xuất file báo cáo | [08-file-formats.md](08-file-formats.md) | §16 |

## Theo chủ đề kỹ thuật

| Chủ đề | File KB | Mục gốc |
|---|---|---|
| Kiến trúc tổng thể | [10-architecture.md](10-architecture.md) | §3, §14, §17 |
| TCP protocol, 10 lệnh | [04-network-protocol.md](04-network-protocol.md) | §8 |
| Latent Factor Model thuật toán | [05-lfm-algorithm.md](05-lfm-algorithm.md) | §7 |
| Khai báo parallel arrays C++ | [06-data-structures.md](06-data-structures.md) | §11.2 |
| ER diagram | [06-data-structures.md](06-data-structures.md) | §11.1 |
| State machine Client | [09-state-machines.md](09-state-machines.md) | §12 |
| State machine Server | [09-state-machines.md](09-state-machines.md) | §13 |

## Theo task thường gặp

| Task | Đọc |
|---|---|
| Thêm một lệnh protocol mới | [04-network-protocol.md](04-network-protocol.md) + [06-data-structures.md](06-data-structures.md) |
| Thêm món vào menu | [03-menu-codes.md](03-menu-codes.md) + [06-data-structures.md](06-data-structures.md) |
| Port LFM từ Python sang C++ | [05-lfm-algorithm.md](05-lfm-algorithm.md) + [matrix_factorization.py](../../matrix_factorization.py) |
| Sửa luồng khách nhập SDT | [02-business-rules.md](02-business-rules.md) (BR11,12,15) + [09-state-machines.md](09-state-machines.md) + [07-ux-cli-design.md](07-ux-cli-design.md) |
| Tính giảm giá 25% | [02-business-rules.md](02-business-rules.md) (BR04) |
| Xuất báo cáo cuối ca | [08-file-formats.md](08-file-formats.md) + [02-business-rules.md](02-business-rules.md) (BR07) |
| Cá nhân hóa gợi ý món | [05-lfm-algorithm.md](05-lfm-algorithm.md) + [02-business-rules.md](02-business-rules.md) (BR14,15) |

## File ngoài KB

- [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) — tài liệu đặc tả gốc 1117 dòng. Là source of truth. Chỉ đọc khi KB thiếu chi tiết.
- [README.md](../../README.md) — hướng dẫn build + run dự án.
- [docs/ML_ENGINE_DESIGN.md](../../docs/ML_ENGINE_DESIGN.md) — thiết kế ML engine (vòng đời model, early stopping, 3 luồng update).
- [matrix_factorization.py](../../matrix_factorization.py) — implementation LFM hoàn chỉnh bằng Python.
- [tools/seed_data.cpp](../../tools/seed_data.cpp) — sinh dữ liệu mẫu (10 personas + ~180 per-order transactions + LFM đã train). Chạy `./build/seed_data` để tạo `data/users.dat`, `data/transactions.dat`, `data/lfm_*.dat`, `data/personas.txt`.
- [tools/dump_data.mjs](../../tools/dump_data.mjs) — Node.js tool parse binary `users.dat` + `transactions.dat` → text readable. Chạy `node tools/dump_data.mjs` từ gốc project. Xem state runtime hiện tại (khác `personas.txt` là snapshot seed).
- [data/personas.txt](../../data/personas.txt) — danh mục 10 SDT mẫu + lịch sử per-order + top-3 gợi ý. Snapshot chỉ sinh bởi `seed_data`; **không update runtime**. Dùng để test với SDT có sẵn (vd `0901234567 = Anh Nam`).
