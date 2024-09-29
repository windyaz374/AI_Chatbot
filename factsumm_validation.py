from factsumm import FactSumm

factsumm = FactSumm()

if __name__ == '__main__':
    article = "TP.Hội An (Quảng Nam) lần thứ 5 được vinh dự nhận được danh hiệu \"Điểm đến thành phố văn hóa hàng đầu châu Á\", trong đó có 4 năm liên tiếp đoạt giải (các năm 2019, 2021, 2022, 2023 và 2024)."
    summary = "Hội An (Quảng Nam) đã được nhận danh hiệu \"Điểm đến thành phố văn hóa hàng đầu châu Á\" 3 lần. Vào các năm 1999, 2004 và 2007."
    factsumm.extract_facts(article, summary, verbose=True)
