from sentence_transformers import SentenceTransformer, util
import json
import torch
import re

# === 1. Load dữ liệu JSON ===
with open("data2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. Load mô hình Sentence Transformer tiếng Việt ===
encoder = SentenceTransformer("keepitreal/vietnamese-sbert")

# === 3. Chuẩn bị danh sách vi phạm ===
violations = []
for v in data["tat_ca_vi_pham"]:
    violations.append({
        "text": v["mo_ta"],  # phần mô tả để encode
        "info": v
    })

# === 4. Encode toàn bộ mô tả (tạo embedding 1 lần) ===
print("Đang tạo embedding mô tả vi phạm...")
violation_embeddings = encoder.encode(
    [v["text"] for v in violations],
    convert_to_tensor=True
)
print(" Đã tạo xong embedding.\n")

# === 5. Hàm phát hiện loại phương tiện ===
def detect_vehicle_type(query: str) -> str | None:
    query = query.lower().strip()
    if re.search(r"\bxe\s*máy\s*chuyên\s*dụng\b", query):
        return "xe_may_chuyen_dung"

    # Ưu tiên mô tô/xe máy trước
    if re.search(r"\bmô\s*tô\b", query) or re.search(r"\bxe\s*máy\b", query) or re.search(r"\bgắn\s*máy\b", query):
        return "xe_moto_xe_may"

    # Sau đó mới xét ô tô
    if re.search(r"\bô\s*tô\b", query) or re.search(r"\boto\b", query) or re.search(r"\bxe\s*hơi\b", query):
        return "xe_oto"

    # Các loại khác
    if re.search(r"\bxe\s*đạp\b", query) or re.search(r"\bxe\s*dap\b", query):
        return "xe_dap"
    if re.search(r"\bngười\s*đi\s*bộ\b", query) or re.search(r"\bđi\s*bộ\b", query):
        return "nguoi_di_bo"

    return None


# === 6. Tiền xử lý câu hỏi (loại bỏ ký tự thừa) ===
def preprocess_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


# === 7. Tìm vi phạm phù hợp ===
def find_violation(query: str):
    query = preprocess_query(query)

    #  Nếu câu quá ngắn hoặc vô nghĩa thì bỏ qua
    if len(query.split()) < 2:
        return None

    vehicle = detect_vehicle_type(query)

    # Lọc theo loại phương tiện
    if vehicle:
        filtered_indices = [
            i for i, v in enumerate(violations)
            if v["info"].get("loai_phuong_tien") == vehicle
        ]
    else:
        filtered_indices = list(range(len(violations)))  # nếu chưa biết loại, dùng toàn bộ

    if not filtered_indices:
        return None

    # Encode câu hỏi
    query_emb = encoder.encode(query, convert_to_tensor=True)

    # Tính cosine similarity
    selected_embeds = violation_embeddings[filtered_indices]
    cos_scores = util.cos_sim(query_emb, selected_embeds)[0]

    # Lấy điểm cao nhất và chỉ số tương ứng
    top_idx = cos_scores.argmax().item()
    top_score = cos_scores[top_idx].item()

    # ⚠Nếu độ tương đồng quá thấp, coi như không khớp
    if top_score < 0.45:
        return None

    selected_violation = violations[filtered_indices[top_idx]]
    return selected_violation["info"]


# === 8. Sinh câu trả lời ===
def answer_violation(query: str) -> str:
    law = find_violation(query)
    if not law:
        return (
            " Xin lỗi, tôi chưa hiểu câu hỏi của bạn.\n"
            " Hãy mô tả rõ hành vi, ví dụ:\n"
            " - 'Xe máy vượt đèn đỏ'\n"
            " - 'Ô tô bóp còi trong khu dân cư'\n"
            " - 'Người đi bộ băng qua đường sai chỗ'"
        )

    ten = law.get("ten_vi_pham", "Không rõ")
    muc = law.get("muc_phat", "Không rõ")
    dieu = law.get("dieu_khoan", "Không rõ")
    tru = law.get("tru_diem", 0)
    mota = law.get("mo_ta", "Không có mô tả.")
    xe = law.get("loai_phuong_tien", "Không rõ")

    return (
        f"**Loại phương tiện:** {xe}\n"
        f"**Hành vi vi phạm:** {ten}\n"
        f"**Điều khoản:** {dieu}\n"
        f"**Mức phạt:** {muc}\n"
        f"**Trừ điểm:** {tru} điểm\n"
        f"**Mô tả:** {mota}"
    )


# === 9. Giao diện console ===
if __name__ == "__main__":
    print("Chatbot Giao Thông Việt Nam (gõ 'exit' để thoát)\n")
    while True:
        q = input("Bạn hỏi: ")
        if q.lower().strip() in ["exit", "quit", "thoát"]:
            print(" Tạm biệt!")
            break
        print(answer_violation(q))
        print("-" * 60)
