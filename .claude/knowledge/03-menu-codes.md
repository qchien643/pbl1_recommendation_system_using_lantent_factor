# 03 · Bảng mã món

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §2.2.

## Quy tắc mã

Mã gồm **3 ký tự ASCII**: `[Prefix][Số thứ tự 2 chữ số]`.

| Prefix | Nhóm |
|---|---|
| `P` | Phở |
| `B` | Bún |
| `C` | Cơm |
| `G` | Gỏi |
| `A` | Ăn vặt |
| `D` | Đồ uống |
| `T` | Tráng miệng |

## Bảng 11 món chuẩn

| Mã | Tên món | Đơn giá |
|---|---|---|
| `P01` | Phở Bò Tái | 65.000đ |
| `P02` | Phở Gà | 55.000đ |
| `B01` | Bún Bò Huế | 60.000đ |
| `B02` | Bún Riêu | 55.000đ |
| `C01` | Cơm Tấm Sườn Bì | 75.000đ |
| `C02` | Cơm Chiên Dương Châu | 65.000đ |
| `G01` | Gỏi Cuốn (5 cuốn) | 55.000đ |
| `A01` | Chả Giò (10 cái) | 80.000đ |
| `D01` | Trà Đá | 15.000đ |
| `D02` | Nước Ngọt | 20.000đ |
| `T01` | Chè Ba Màu | 25.000đ |

## Format file `data/menu.txt`

Mỗi dòng 1 món: `MA|TEN|GIA|CATEGORY`

```
P01|Pho Bo Tai|65000|P
P02|Pho Ga|55000|P
B01|Bun Bo Hue|60000|B
...
```

## Validate code trong C++

```cpp
bool isValidMenuCode(const char* code) {
    if (strlen(code) != 3) return false;

    char prefix = code[0];
    if (prefix != 'P' && prefix != 'B' && prefix != 'C' &&
        prefix != 'G' && prefix != 'A' && prefix != 'D' && prefix != 'T')
        return false;

    if (!isdigit(code[1]) || !isdigit(code[2])) return false;

    // Tra cứu trong menuCode[] xem có tồn tại không
    for (int i = 0; i < menuCount; i++)
        if (strcmp(menuCode[i], code) == 0) return true;

    return false;
}
```

Xem array `menuCode[MAX_MENU][4]` trong [06-data-structures.md](06-data-structures.md).

## Khi thêm món mới

1. Thêm dòng vào `data/menu.txt`.
2. Tăng `MAX_MENU` nếu > 20 (mặc định).
3. Khởi tạo `Q[new_i][K]` ngẫu nhiên nhỏ + chạy `add_items()` flow — xem [05-lfm-algorithm.md](05-lfm-algorithm.md).
4. Broadcast `MENU_DATA` lại tới tất cả Client đang kết nối.
