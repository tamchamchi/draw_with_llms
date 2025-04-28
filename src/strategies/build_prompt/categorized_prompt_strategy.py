from typing import Dict, List, Optional, Tuple

from ..base import PromptBuildingStrategy

KEYWORD_BANK = {
    "fashion": [
        # Top (Áo - Phần trên & Áo khoác)
        "shirt", "blouse", "t-shirt", "top", "sweater", "knitwear", "hoodie",
        "vest", "jacket", "coat", "blazer", "cardigan",

        # Bottom (Quần - Phần dưới)
        # Overalls bao gồm cả phần dưới
        "pants", "trousers", "jeans", "skirt", "shorts", "overalls",

        # Full Body / Other Garments (Đồ liền thân / Loại khác)
        "dress", "suit", "jumpsuit", "romper", "swimwear", "lingerie", "gown",
        "costume", "uniform",

        # Accessories (Phụ kiện)
        "accessory", "accessories", "bag", "handbag", "purse", "clutch", "backpack",
        "belt", "hat", "cap", "beanie", "scarf", "gloves", "sunglasses", "watch",
        "jewelry", "necklace", "earrings", "bracelet", "ring", "tie",

        # Shoes (Giày)
        "shoes", "footwear", "boots", "heels", "sneakers", "trainers", "sandals", "flats",

        # Material (Chất liệu)
        "fabric", "textile", "material", "cotton", "silk", "wool", "leather", "denim",
        "linen", "velvet", "satin", "lace", "chiffon", "polyester", "nylon", "rayon",
        "cashmere", "tweed", "canvas", "corduroy", "faux fur",

        # Detail (Chi tiết)
        "buttons", "zipper", "pocket", "pockets", "collar", "sleeve", "cuff", "hem",
        "lapel", "embroidery", "sequins", "beads", "applique", "patch", "tassel",
        "trim", "print", "pattern", "logo", "ruffles", "pleats", "studs",

        # phong cách (Style / Aesthetic)
        "style", "vintage", "retro", "minimalist", "maximalist", "bohemian", "chic",
        "elegant", "casual", "formal", "streetwear", "sporty", "avant-garde",
        "classic", "trendy", "preppy", "grunge", "punk", "romantic", "dreamy",

        # đặc điểm (Characteristics / Features / Concepts)
        "texture", "fit", "silhouette", "tailored", "bespoke", "custom", "oversized",
        "design", "designer", "collection", "runway", "catwalk", "lookbook",
        "trend", "wardrobe", "clothing", "apparel", "garment", "wear", "outfit",
    ],

    "abstract": [
        # Shapes / Geometry (Hình khối / Hình học)
        "rectangles", "pyramids", "cone", "trapezoids", "sheet", "triangles",
        "crescents", "dodecahedron", "geometry", "shape", "organic",
        "squares", "circles",  # Thêm các hình cơ bản khác nếu cần

        # Composition / Structure (Bố cục / Cấu trúc)
        "grid", "spiraling", "layered", "interwoven", "pattern", "balance",
        "fragmented", "distorted", "floating", "composition", "structure",

        # Style / Movement (Phong cách / Trường phái)
        "surreal", "conceptual", "minimal", "cubism", "nonrepresentational",
        "expressionism",  # Có thể thêm các trường phái khác

        # Concepts / Feelings (Khái niệm / Cảm giác)
        "chaotic", "formless", "dreamlike", "expression", "illusion", "visual",
        "motion", "energy", "ambiguous", "random", "chaos", "imaginary",
        "symbolic", "harmony", "dynamic",

        # Technique / Elements (Kỹ thuật / Yếu tố)
        "threads", "colors", "texture", "stroke", "blurred", "vivid", "brush",
    ],

    "landscapes": [
        # Geographic Features (Land) - Đặc điểm địa lý (Đất liền)
        "plain",        # Đồng bằng
        "peaks",        # Đỉnh núi
        "mountain",     # Núi
        "valley",       # Thung lũng
        "hill",         # Đồi
        "cliff",        # Vách đá
        "sand",         # Cát
        "desert",       # Sa mạc
        "glacier",      # Sông băng
        "rocks",        # Đá
        "terrain",      # Địa hình

        # Geographic Features (Water) - Đặc điểm địa lý (Nguồn nước)
        "ocean",        # Đại dương
        "lagoon",       # Đầm phá
        "lake",         # Hồ
        "river",        # Sông
        "coast",        # Bờ biển (nói chung)
        "shore",        # Bờ (sông, hồ, biển)
        "bay",          # Vịnh
        "beach",        # Bãi biển
        "sea",          # Biển

        # Atmospheric / Sky / Time / Weather - Khí quyển / Bầu trời / Thời gian / Thời tiết
        "dusk",         # Hoàng hôn
        "cloudy",       # Nhiều mây
        "sky",          # Bầu trời
        "snowy",        # Có tuyết
        "sunset",       # Lúc mặt trời lặn
        "sunrise",      # Lúc mặt trời mọc
        "dawn",         # Bình minh
        "starlit",      # Đầy sao
        "night",        # Ban đêm
        "mist",         # Sương mù nhẹ
        "fog",          # Sương mù dày
        "horizon",      # Đường chân trời
        "wind",         # Gió
        "weather",      # Thời tiết
        "atmospheric",  # Thuộc về khí quyển, tạo không khí

        # Flora / Vegetation - Hệ thực vật
        "forest",       # Rừng
        "trees",        # Cây cối
        "meadow",       # Đồng cỏ
        "field",        # Cánh đồng
        "flora",        # Hệ thực vật (từ chung)
        "vegetation",   # Thảm thực vật

        # Man-Made Elements - Yếu tố nhân tạo
        "lighthouse",   # Hải đăng
        "trail",        # Đường mòn
        "path",         # Lối đi
        "skyline",      # Đường chân trời (thường là của thành phố)
        "cityscape",    # Cảnh quan thành phố

        # General / Concepts - Khái niệm chung
        "nature",       # Thiên nhiên
        "scenery",      # Phong cảnh, cảnh vật
        "countryside",  # Miền quê
        "view",         # Tầm nhìn, quang cảnh
        "panorama",     # Toàn cảnh
        "landscape",    # Phong cảnh (từ chung)
        "natural",      # Thuộc về tự nhiên
        "environment",  # Môi trường
        "wild",         # Hoang dã
        "outdoor",      # Ngoài trời
        "lighting",     # Ánh sáng (có thể coi là đặc điểm)
    ],
}

CATEGORY_PROMPTS = {
    "fashion": {
        "prefix": "Minimal illustration, flat vector icon, nostalgic feeling, soft warm light, of",
        "suffix": "with high contrast, watercolor illustration, romantic and dreamy mood, dreamy lighting, impressionist atmosphere.",
        "negative": "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients",
    },
    "landscapes": {
        "prefix": "Simple, Classic, Minimal illustration, ((flat vector icon)) of",
        "suffix": "with flat color blocks, high contrast, beautiful, outline only, no details, solid color only.",
        "negative": "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients",
    },
    "abstract": {
        "prefix": "Simple, Classic, Minimal illustration, flat vector icon, nostalgic feeling, soft warm light, wet-on-wet of",
        "suffix": "with flat color blocks, high contrast, watercolor illustration, romantic and dreamy mood.",
        "negative": "",
    },
    "general": {  # Default / Fallback
        "prefix": "Minimal illustration, flat vector icon, nostalgic feeling, soft warm light, of",
        "suffix": "with high contrast, watercolor illustration, romantic and dreamy mood, dreamy lighting, impressionist atmosphere.",
        "negative": "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients",
    },
}


class CategorizedPromptStrategy(PromptBuildingStrategy):
    """
    Chiến lược xây dựng prompt dựa trên việc phân loại description
    và gọi hàm build tương ứng cho từng loại.
    """

    def __init__(self, keyword_bank: Dict[str, List[str]] = KEYWORD_BANK):
        self.keyword_bank = keyword_bank
        self.category_order = ["fashion", "landscapes", "abstract"]
        print(
            "--- Prompt Strategy [Categorized]: Initialized (using specific build functions). ---"
        )

    def _classify_description(self, description: str) -> str:
        """Phân loại description dựa vào số lượng keyword khớp nhiều nhất."""
        if not description:
            print("Classification: Empty description -> Category='general'")
            return "general"

        desc_lower = description.lower()
        scores = {}

        # Tính điểm cho các category có trong bank (có thể bỏ qua 'general' nếu nó có trong bank)
        categories_to_score = [
            cat for cat in self.keyword_bank.keys() if cat != 'general']

        if not categories_to_score:
            print(
                "Classification: No specific categories found in keyword bank -> Category='general'")
            return "general"

        for category in categories_to_score:
            keywords = self.keyword_bank.get(category, [])  # Lấy list keywords
            # Tính tổng số lần khớp keyword (đếm số keyword có trong description)
            scores[category] = sum(kw.lower() in desc_lower for kw in keywords)

        print(f"Classification Scores: {scores}")  # In điểm số (để debug)

        # Tìm điểm cao nhất
        max_score = 0
        if scores:  # Đảm bảo scores không rỗng
            try:
                max_score = max(scores.values())
            # Trường hợp scores chỉ chứa giá trị không hợp lệ (dù không nên xảy ra)
            except ValueError:
                max_score = 0

        # Nếu không có keyword nào khớp (tất cả score là 0) -> general
        if max_score == 0:
            print("Classification: All scores are 0 -> Category='general'")
            return "general"  # <<< Trả về "general" thay vì "none"

        # Tìm các category đạt điểm cao nhất (có thể hòa điểm)
        top_categories = [cat for cat,
                          score in scores.items() if score == max_score]

        if len(top_categories) == 1:
            best_category = top_categories[0]
            print(
                f"Classification: Top score {max_score} -> Category='{best_category}'")
            return best_category
        else:
            # Xử lý hòa điểm: Ưu tiên theo self.category_order
            print(
                f"Classification: Tie detected between {top_categories} with score {max_score}.")
            for cat_in_order in self.category_order:
                if cat_in_order in top_categories:
                    print(
                        f"Classification: Resolving tie using order -> Category='{cat_in_order}'")
                    return cat_in_order
            # Nếu không category nào trong order bị hòa, trả về cái đầu tiên
            best_category = top_categories[0]
            print(
                f"Classification: Resolving tie by taking first -> Category='{best_category}'")
            return best_category

    # --- Các hàm build riêng cho từng category (có thể là private) ---

    def _build_fashion_prompt(self, description: str) -> Tuple[str, str]:
        prefix_prompt = "icon vector graphic, focus on"
        suffix_prompt = "with minimalist design, flat color illustration"
        negative_prompt = "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients"
        prompt = f"{prefix_prompt} (({description})) {suffix_prompt}".strip()
        return prompt, negative_prompt

    def _build_landscapes_prompt(self, description: str) -> Tuple[str, str]:
        prefix_prompt = "Simple, Classic, Minimal illustration, ((flat vector icon)) of"
        suffix_prompt = "with flat color blocks, high contrast, beautiful, outline only, no details, solid color only."
        negative_prompt = "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients"
        prompt = f"{prefix_prompt} {description} {suffix_prompt} Negative: {negative_prompt}".strip()
        return prompt, negative_prompt

    def _build_abstract_prompt(self, description: str) -> Tuple[str, str]:
        prefix_prompt = "Simple, Classic, Minimal illustration, flat vector icon, nostalgic feeling, soft warm light, wet-on-wet of"
        suffix_prompt = "with flat color blocks, high contrast, watercolor illustration, romantic and dreamy mood."
        negative_prompt = ""
        prompt = f"{prefix_prompt} {description} {suffix_prompt}".strip()
        return prompt, negative_prompt

    def _build_general_prompt(self, description: str) -> Tuple[str, str]:
        prefix_prompt = "Simple, Classic, Minimal illustration, ((flat vector icon)) of"
        suffix_prompt = "with flat color blocks, high contrast, beautiful, outline only, no details, solid color only."
        negative_prompt = "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients"
        prompt = f"{prefix_prompt} {description} {suffix_prompt}".strip()
        return prompt, negative_prompt

    # --- Phương thức build chính ---
    def build(self, description: str, **kwargs) -> Dict[str, Optional[str]]:
        """Xây dựng prompt bằng cách gọi hàm tương ứng với category."""
        category = self._classify_description(description)
        print(f"    Building prompt for category: '{category}'")

        # Gọi hàm build tương ứng
        if category == "fashion":
            positive_prompt, negative_prompt = self._build_fashion_prompt(
                description)
        elif category == "landscapes":
            positive_prompt, negative_prompt = self._build_landscapes_prompt(
                description
            )
        elif category == "abstract":
            positive_prompt, negative_prompt = self._build_abstract_prompt(
                description)
        else:  # Mặc định là general
            positive_prompt, negative_prompt = self._build_general_prompt(
                description)

        # Trả về dictionary theo yêu cầu của interface
        return {"prompt": positive_prompt, "negative_prompt": negative_prompt}
