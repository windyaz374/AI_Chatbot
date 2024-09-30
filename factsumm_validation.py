import bert_score
from factsumm import FactSumm

factsumm = FactSumm()

if __name__ == '__main__':
    articles = [
        "Bệnh ung thư trực tràng có khả năng chữa khỏi cao nếu được phát hiện và điều trị sớm",
        "Hành khách đi chân trần trên máy bay ngày càng đông nhưng không phải tất cả các hãng hàng không đều chấp nhận, nhiều hãng Mỹ cấm bay nếu khách không đi tất",
        "Tế bào bạch cầu (WBC) là một phần của hệ thống miễn dịch, với nhiệm vụ bảo vệ cơ thể khỏi các tác nhân gây nhiễm trùng. Số lượng tế bào bạch cầu cao là dấu hiệu cho thấy cơ thể đang chống lại nhiễm trùng. Số lượng tế bào bạch cầu thấp có thể do một số loại thuốc hoặc nhiễm trùng gây ra",
        "Tổ chức Y tế Thế giới (WHO) ước tính, mỗi năm trên thế giới có đến 17,5 triệu người tử vong do các bệnh lý liên quan đến tim mạch",
        "Tắc nghẽn động mạch là kết quả của quá trình xơ vữa động mạch, khi các mảng bám tích tụ trên thành động mạch gây hẹp hoặc tắc nghẽn lòng mạch",
        "trong năm 2022, thế giới có khoảng 120.434 ca mới mắc ung thư vòm họng",
        "Sổ mũi là hiện tượng chất nhầy hoặc chất lỏng chảy ra từ mũi, thường có dạng nước hoặc đặc hơn, trong suốt hoặc đục",
        "Nếu nước mũi của trẻ có màu trắng trong, phụ huynh có thể nhỏ nước muối NaCl 0,9%, mỗi bên 3 – 4 giọt, 4 – 5 lần mỗi ngày",
        "Thoát vị bẩm sinh xảy ra ở khoảng 15% trẻ sơ sinh",
        "Huyết áp cao được cho là một trong những yếu tố nguy cơ chính dẫn đến đột quỵ. Hầu hết những người bị đột quỵ lần đầu được ghi nhận là có mắc bệnh huyết áp cao",
        "Với diện tích chỉ 2km², Monaco là quốc gia nhỏ thứ nhì thế giới",
        "dịp lễ Quốc khánh 2.9, Bình Định đón 204.044 lượt du khách",
        "Một số điểm du lịch nổi bật như Eo Gió, Kỳ Co, Ghềnh Ráng (TP.Quy Nhơn)",
        "Dự án Ninh Chữ Sailing Bay được nhà đầu tư bấm nút khởi công vào ngày 9.12.2021",
        "Cảng hàng không quốc tế Cam Ranh dự kiến dịp lễ 2-9 có khoảng 450 lượt chuyến bay cất/hạ cánh xuống sân bay quốc tế Cam Ranh",
        "Cách đây 110 năm, bác sĩ Alexandre Yersin đã xây dựng một ngôi nhà gỗ trên núi Hòn Bà, qua thời gian ngôi nhà bị xuống cấp, song đến thời điểm hiện tại ngôi nhà gỗ của vị bác sĩ lừng danh đã được một doanh nghiệp du lịch tu bổ, để phục vụ du khách thập phương",
        "tỉ phú Ấn Độ Sun Pharmaceutical Industries Limited sẽ đưa khoảng 4.500 nhân viên đến Việt Nam du lịch trong khoảng thời gian từ ngày 27-8 đến 3-9",
        "Năm 2023, Hà Giang đã được vinh danh là Điểm đến du lịch mới nổi hàng đầu châu Á",
        "Hãng hàng không Vietravel (Vietravel Airlines) lần thứ hai liên tiếp được vinh danh là Hãng hàng không có trải nghiệm dành cho du lịch hàng đầu châu Á",
        "Núi Bà Đen nổi lên giữa vùng đồng bằng với độ cao 986m",
        "Trong bình chọn 100 món xào ngon nhất, trang chuyên về ẩm thực Taste Atlas xếp rau muống xào tỏi ở số 17 thế giới",
        "Sụp mí mắt (Ptosis) là tình trạng mí mắt trên bị chảy xệ quá mức. Mép của mí mắt trên thấp hơn mức bình thường hoặc do vấn đề về da liễu như có da thừa ở mí mắt trên",
        "Nguyên nhân sụp mí mắt có thể do: chấn thương, tác dụng phụ của thuốc điều trị, bệnh tật, bẩm sinh hoặc các nguyên nhân tự nhiên khác",
        "Omicron vốn là một biến thể của chủng virus Sars Cov-2. Trong những tuần gần đây, xuất hiện một biến thể phụ Omicron mới có tên gọi là XBB hoặc là BA.2.10. Biến chủng XBB Omicron là nguyên nhân đang gây bùng lên đợt dịch COVID-19 mới tại Singapore – trong vòng 3 tuần đã phát triển mạnh chiếm hơn 1⁄2 số ca nhiễm mỗi ngày tại đây",
        "kháng thể đơn dòng Evusheld ngừa Covid-19 giúp giảm 83% nguy cơ mắc Covid-19",
    ]
    summaries = [
        "Ung thư trực tràng có thể điều trị được, nhưng hiệu quả của điều trị phụ thuộc vào giai đoạn và loại ung thư",
        "Không, hành khách không được phép đi chân trần trên máy bay",
        "Tế bào bạch cầu là một loại tế bào máu trắng, có chức năng bảo vệ cơ thể chống lại các tác nhân alien và virus khác. Tế bào bạch cầu được sản xuất trong tủy xương và di chuyển vào máu để thực hiện chức năng của mình. Tế bào bạch cầu đóng vai trò rất quan trọng trong việc bảo vệ cơ thể chống lại bệnh tật. Nếu số lượng tế bào bạch cầu thấp hoặc không đủ, sẽ làm tăng nguy cơ mắc bệnh và thậm chí có thể dẫn đến tử vong",
        "Theo Tổ chức Y tế Thế giới (WHO), khoảng 17,5 triệu người tử vong mỗi năm do các bệnh lí về tim mạch",
        "Tắc nghẽn động mạch (thrombosis) là tình trạng hình thành một cục máu đông (thrombus) trong lòng động mạch, ngăn chặn dòng chảy của máu và gây ra thiếu máu cho các mô xung quanh",
        "Theo Tổ chức Y tế Thế giới (WHO), ước tính có khoảng 640.000 ca mới mắc ung thư vòm họng toàn cầu trong năm 2022",
        "Sổ mũi (Nasal Polyp) là một loại polyp (dạng khối u) hình thành trong mũi do các tế bào epithelial và các tế bào máu trưởng thành không bình thường. Sổ mũi có thể gây ra nhiều vấn đề về hô hấp",
        "Nước muối (Saline Solution) là một dung dịch chứa muối và nước, được sử dụng để điều trị các vấn đề về hô hấp, bao gồm sổ mũi. Khi nhỏ nước muối vào mũi, dung dịch này có thể giúp làm loãng chất tiết ra do sổ mũi, giảm thiểu tình trạng nghẽn đường thở và cải thiện chức năng hô hấp",
        "Theo các tài liệu y khoa, tỷ lệ xảy ra thoát vị bẩm sinh (Congenital Hernia) là khoảng 1-5%",
        "Theo các nghiên cứu, tăng huyết áp được xem là một yếu tố nguy cơ quan trọng cho đột quỵ, đặc biệt ở người lớn tuổi. Do đó, điều trị tăng huyết áp kịp thời và hiệu quả là rất quan trọng để giảm thiểu rủi ro đột quỵ",
        "Quốc gia nhỏ thứ nhì thế giới là Monaco, có diện tích khoảng 2km²",
        "lễ Quốc khánh 2/9 tại tỉnh Bình Định đã đón hơn 200.000 lượt khách tham gia",
        "Các điểm du lịch nổi bật ở Bình Định bao gồm: Cầu Độc Lập, Pháo đài Quần Linh, Lăng Chủ tịch Hồ Chí Minh, đền thờ Bà Chúa Ngãi, Quảng trường Hoà Phát, Động Phong Nha-Kẻ Bàng",
        "Dự án Ninh Chữ Sailing Bay đã được chính quyền tỉnh Bình Định và các đơn vị liên quan phê duyệt, và dự án này đã được khởi công từ ngày 26 tháng 2 năm 2016",
        "Dự kiến, trong dịp lễ Quốc khánh 2-9, cảng hàng không quốc tế Cam Ranh sẽ có khoảng 200 chuyến bay đến từ các nước và miền đất",
        "Nơi làm việc của bác sĩ Alexandre Yersin trên đỉnh Hòn Bà là một trạm y tế được xây dựng vào năm 1895, thuộc tỉnh Bình Định. Trạm y tế này là nơi bác sĩ Alexandre Yersin, người Pháp, đã làm việc và nghiên cứu về bệnh dịch hạch, kết quả là phát hiện ra nguyên nhân gây bệnh dịch hạch là vi khuẩn Pasteurella pestis",
        "Ngày 25 tháng 5 năm 2019, một đoàn du khách gồm 14 người từ Ấn Độ đã đến thăm Bình Định. Đoàn du khách này được dẫn bởi Công ty Dược phẩm Sun Pharmaceutical Industries Limited, một trong những công ty dược phẩm lớn nhất Ấn Độ",
        "Hà Giang được vinh danh là 'Điểm đến du lịch mới nổi hàng đầu châu Á' năm 2019",
        "Hãng hàng không Vietravel (Vietravel Airlines) được vinh danh là Hãng hàng không có trải nghiệm dành cho du lịch hàng đầu châu Á 2 lần",
        "Độ cao của Núi Bà Đen là 986 mét",
        "Rau muống xào tỏi của Taste Atlas xếp hạng 17 trong top 100 món xào ngon nhất",
        "Sụp mí mắt (ptosis) là một tình trạng y tế xảy ra khi mí mắt của người bệnh hạ xuống quá mức, thường do vấn đề với dây thần kinh hoặc cơ mí",
        "Nguyên nhân của sụp mí mắt (ptosis) có thể là do độc tơ thần kinh, bệnh thần kinh, ung thư não, đột quất, bệnh mỡ, phẫu thuật, trào ngược vị giác: Sụp mí mắt có thể xảy ra do trào ngược vị giác và các vấn đề về dạ dày",
        "Trong Covid-19, biến thể Omicron là một trong những phiên bản mới nhất của virus SARS-CoV-2 được phát hiện vào tháng 11 năm 2021. Biến thể này được cho là có tỷ lệ lây truyền cao hơn và có thể gây ra các triệu chứng nặng hơn so với các phiên bản trước",
        "Kháng thể đơn dòng Evusheld có thể giúp giảm nguy cơ mắc COVID-19 khoảng 70% đến 85%",
    ]

    for idx, (article, summary) in enumerate(zip(articles, summaries), start=1):
        print(f"\nProcessing article {idx}")
        P, R, F1 = bert_score.score([article], [summary], lang="vi")
        print(f"Article {idx} - BERTScore: {F1}")

    P, R, F1 = bert_score.score(summaries, articles, lang='vi')
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1 Score: {F1.mean().item():.4f}")

