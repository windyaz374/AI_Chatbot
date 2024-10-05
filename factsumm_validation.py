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

        "Trong tháng 8/2024, Việt Nam đón 1,43 triệu lượt khách quốc tế, tăng 17,7 phần trăm so với cùng kỳ năm trước", 
        "Các tour Đông Bắc Á (Nhật Bản, Hàn Quốc, Đài Loan) được giảm 2 triệu đồng",
        "Nước dùng thơm ngon, được ninh từ xương bò, kết hợp với nhiều nguyên liệu như chân giò heo, gân bò và chả cua",
        "Tăng 21,2%", 
        "Vé máy bay đắt đỏ cùng tình trạng trễ chuyến xảy ra thường xuyên",
        "Hai địa điểm, một điểm bắn tầm cao ở khu vực đầu đường hầm sông Sài Gòn, TP Thủ Đức và một điểm bắn tầm thấp ở công viên văn hóa Đầm Sen, quận 11",
        "Từ 7h - 17h30",
        "Có bệnh viện, khu nghỉ dưỡng trượt tuyết, khách sạn, trung tâm thương mại, văn hóa.",
        "Quốc gia hạnh phúc nhất thế giới dành cho những người dưới 30 tuổi (gen Z) theo Báo cáo Hạnh phúc Thế giới (World Happiness Report) 2024 của Liên Hợp Quốc.", 
        "Người Việt đã có mặt trên đảo Lý Sơn vào cuối thế kỷ 16 đầu thế kỷ 17",
        "Các chỉ số xét nghiệm máu quan trọng bạn cần lưu ý, gồm: WBC, RBC, PLT", 
        "Có nhiều loại rối loạn đông máu nhưng phổ biến gồm Hemophilia, bệnh von Willebrand, chảy máu liên quan đến bệnh gan hoặc do thiếu vitamin K, suy giảm lượng tiểu cầu trong máu",
        "Huyết áp cao làm cho động mạch dễ bị vỡ hoặc tắc nghẽn, gây nguy cơ đột quỵ cao hơn",
        "Nghiên cứu gần đây ước tính lượng muối trung bình tiêu thụ ở Việt Nam là 9,4 g/ngày, gần gấp đôi mức 5g/ngày do Tổ chức Y tế Thế giới (WHO) khuyến nghị",
        "Người cận từ 6 độ trở lên được xem như cận nặng, cần đeo kính hầu hết thời gian trong ngày",
        "Tình trạng loạn thị xảy ra do bất thường về giác mạc, lúc ấy, ánh sáng không tập trung đều trên võng mạc khiến người bệnh không nhìn rõ vật thể",
        "Ngăn chặn vòng sinh sản của muỗi, tiêm ngừa vắc xin, và duy trì vệ sinh môi trường sống để giảm nguy cơ mắc bệnh", 
        "Tế bào gốc toàn năng (totipotent stem cells TSCs) là loại tế bào gốc linh hoạt nhất, có thể biệt hóa để hình thành toàn bộ phôi thai hoàn chỉnh",
        "HPV lây nhiễm chủ yếu qua quan hệ tình dục (âm đạo, hậu môn hoặc miệng) nhưng cũng có thể lây từ mẹ sang con, qua thủ thuật y tế không đảm bảo vô khuẩn, hoặc tiếp xúc trực tiếp với bộ phận sinh dục",
        "Đa số nam giới nhiễm HPV không có triệu chứng rõ ràng. Tuy nhiên, trong một số trường hợp, có thể quan sát thấy dấu hiệu như mụn cóc sinh dục trên bộ phận sinh dục",
        "Bệnh sởi (Measles) là bệnh nhiễm trùng cấp tính do virus Polinosa Morbillarum gây ra, thường gây sốt, phát ban và có khả năng truyền nhiễm. Bệnh rất dễ lây lan qua không khí khi người bệnh nói chuyện, ho, hắt hơi.",
        "Bất kỳ ai, đặc biệt là người chưa tiêm vaccine phòng sởi, đều có nguy cơ mắc bệnh. Các nhóm dễ mắc hơn bao gồm trẻ em, người lớn tuổi, phụ nữ mang thai và những người thường xuyên tiếp xúc với người bệnh",
        "Các triệu chứng nghi ngờ bao gồm loạng choạng, hoa mắt, chóng mặt, ù tai, và buồn nôn. Nếu có dấu hiệu này, nên đến gặp bác sĩ để được tư vấn điều trị kịp thời",
        "Bệnh động kinh có thể xảy ra ở trẻ em ở mọi lứa tuổi, từ khi sinh ra đến tuổi vị thành niên",
        "Giữ bình tĩnh, tạo không gian an toàn, nới lỏng quần áo, đặt trẻ nằm nghiêng, không cố gắng khống chế sự cử động của trẻ, và không cho trẻ ăn uống gì khi chưa hoàn toàn tỉnh táo"
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

        "Theo thống kê từ Tổng cục Du lịch Việt Nam, số lượng khách quốc tế đến Việt Nam trong tháng 8/2024 là khoảng 420.000 người, tăng tăng 10,5 phần trăm so với tháng 8/2023", 
        "Tour Đông Bắc Á được giảm 2 triệu đồng tại hội chợ Hội chợ ITE HCMC 2024",
        "Món bún bò Huế tại Đà Nẵng là một phiên bản của món ăn nổi tiếng này, được chế biến theo kiểu Huế nhưng có thể khác biệt về hương vị và cách làm so với những quán ăn chuyên về món này tại Huế",
        "Theo các báo cáo về du lịch, lượng khách du lịch đến Đà Nẵng dịp lễ 2.9 thường tăng so với cùng kỳ năm trước. Tuy nhiên, con số chính xác tăng bao nhiêu không được công bố công khai",
        "Việc đi du lịch bằng đường bộ còn mang lại nhiều lợi ích cho gia đình như tiết kiệm chi phí, có thể lựa chọn các điểm đến gần và đảm bảo an toàn hơn so với phương tiện bay",
        "Trong dịp Quốc khánh, Thành phố Hồ Chí Minh thường tổ chức bắn pháo hoa tại hai địa điểm chính là: Sông Sài Gòn (Sài Gòn River) và Công viên Đâm Sen (Đầm Sen Park)",
        "Theo thông tin của di tích Huế, du khách tham quan di tích vào ngày Quốc khánh sẽ được miễn phí trong khoảng thời gian từ 8h00 đến 17h00",
        "Thành phố Samjiyon, Triều Tiên là một địa điểm du lịch được chú trọng phát triển, ở đây có hầm rượu, một số bất động sản của cơ quan y tế, đường sắt, kiểm lâm, vận tải đường biển, đài quan trắc...",
        "Theo báo cáo, Lithuania được xếp hạng là quốc gia hạnh phúc nhất thế giới bởi báo cáo Hạnh phúc Thế giới (World Happiness Report)",
        "Theo các ghi chép lịch sử, người Việt đã có mặt trên đảo Lý Sơn (Quảng Ngãi) từ thời điểm thế kỷ 17",
        "Xét nghiệm máu là một công cụ quan trọng giúp các bác sĩ và y tá theo dõi tình trạng sức khỏe của bệnh nhân, đặc biệt là sau khi điều trị. Các chỉ số quan trọng trong xét nghiệm máu bao gồm: RBC, WBC, PC, BUN,...",
        "Các loại rối loạn đông máu phổ biến bao gồm: Rối loạn đông máu huyết động (VTE), Rối loạn đông máu ngoại vi (DVT), Tắc nghẽn động mạch não (Cerebral thrombosis), tắc nghẽn động mạch tim (Myocardial thrombosis)",
        "Khi huyết áp cao, máu chảy mạnh và gây ra sự co thắt mạch máu, dẫn đến hình thành cục máu đông trong não. Điều này làm giảm lưu lượng máu đến não, gây ra các triệu chứng của đột quỵ như mất chức năng, yếu tay chân, khó nói,...",
        "Theo Cục Quản lý Thực phẩm và Dược phẩm Việt Nam (Cục Quản lý Thực phẩm), người dân Việt Nam tiêu thụ khoảng 9,5 gam muối mỗi ngày",
        "Theo thông tin y tế, độ cận nặng nhất là độ cận -2.50 D đến -3.25 D (cận nặng). Còn độ cận nhẹ hơn -6.00 D",
        "Loạn thị được mô tả là một loại viêm kết mạc ảnh hưởng đến 1 mắt, gây ra các triệu chứng như nốt sạm nhỏ trong tròng trắng mắt và nổi hạch trước tai",
        "Phòng ngừa sốt xuất huyết bao gồm: Vacciination (tiêm vaccine chống sốt xuất huyết). Sử dụng áo màng, váy, và giày để không được côn trùng đốt. Không ngủ ngoài trời hoặc trong khu vực có các con côn trùng gây bệnh. Đừng để nước và thức ăn bị ô nhiễm",
        "Tế bào gốc toàn năng là loại tế bào trong cơ thể con người có khả năng phát triển thành các loại tế bào khác nhau trong cơ thể, bao gồm tế bào máu, tế bào gan, tế bào thận,... và có thể được sử dụng để điều trị nhiều bệnh tật.",
        "HPV có thể được truyền qua giao hợp, quan hệ tình dục và thậm chí là quan hệ tình dục không an toàn (không sử dụng bao cao su). Ngoài ra, người bị nhiễm HPV cũng có thể truyền virus cho gia đình mình thông qua tiếp xúc trực tiếp với những người đã bị nhiễm virus này",
        "Nam giới nhiễm HPV đa số không có triệu chứng rõ ràng",
        "Sởi là bệnh nhiễm trùng do virus varicella-zoster gây ra, đặc biệt ảnh hưởng đến trẻ em và người lớn chưa được tiêm chủng. Bệnh này thường xảy ra vào mùa xuân và thu, với các triệu chứng bao gồm đau rát da, mụn nước và sốt. Sởi có thể lan truyền qua tiếp xúc trực tiếp với người bị bệnh hoặc những người đã mắc bệnh trước đó",
        "Trẻ em dưới 5 tuổi, người lớn trên 65 tuổi, phụ nữ mang thai và người có hệ thống miễn dịch suy yếu",
        "Triệu chứng cần lưu ý để nhận biết rối loạn tiền đình bao gồm: Sốt cao đột ngột và khó kiểm soát, buồn nôn, đổ mồ hôi nhiều, hơi thở nhanh, tim đập mạnh",
        "Nguy cơ bị động kinh có thể xảy ra đối với bất kỳ ai, không phân biệt tuổi tác, giới tính hay tình trạng sức khỏe",
        "Di chuyển ngượi bị động kinh đến một vị trí an toàn, tránh xa nguy hiểm và không có vật sắc nhọn. Nếu bệnh nhân còn tỉnh táo, hãy giúp họ ngồi xuống và cố gắng làm yên tâm. Giúp bệnh nhân nằm ngửa. Gọi cấp cứu gần nhất",
    ]

    for idx, (article, summary) in enumerate(zip(articles, summaries), start=1):
        print(f"\nProcessing article {idx}")
        P, R, F1 = bert_score.score([article], [summary], lang="vi")
        print(f"Article {idx} - BERTScore: {F1}")

    P, R, F1 = bert_score.score(summaries, articles, lang='vi')
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1 Score: {F1.mean().item():.4f}")

