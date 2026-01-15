# -*- coding: utf-8 -*-

test_prompts = [
    # ---------------- ENGLISH ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "Analyze sentiment from the latest quarterly earnings report.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "What's the sentiment regarding Tesla's new product launch?", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Determine the emotional tone of this market analysis report.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Is this financial news article positive or negative about Apple?", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "Classify this business news headline into its correct sector.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Categorize this financial article by industry type.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "What category does this economic report belong to?", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Classify this market update by financial sector.", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "Explain this concept in simple terms.", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "What is the difference between RAM and storage?", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Summarize the main points in this research article.", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "Help me understand why this error occurs.", 'domain': 'general', 'task': 'question_answering'},

    # ---------------- GERMAN ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "Analysiere die Stimmung im neuesten Quartalsbericht.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Wie ist die Stimmung bezüglich der neuen Produkteinführung von Tesla?", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Bestimme den emotionalen Ton dieses Marktanalyseberichts.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Ist dieser Finanznachrichtenartikel über Apple positiv oder negativ?", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "Ordne diese Schlagzeile der Wirtschaftsnachrichten der richtigen Branche zu.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Kategorisiere diesen Finanzartikel nach Industrietyp.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Zu welcher Kategorie gehört dieser Wirtschaftsbericht?", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Klassifiziere dieses Marktupdate nach Finanzsektor.", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "Erkläre dieses Konzept in einfachen Worten.", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Was ist der Unterschied zwischen RAM und Speicher?", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Fasse die Hauptpunkte dieses Forschungsartikels zusammen.", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "Hilf mir zu verstehen, warum dieser Fehler auftritt.", 'domain': 'general', 'task': 'question_answering'},

    # ---------------- SPANISH ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "Analiza el sentimiento del último informe trimestral.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "¿Cuál es el sentimiento sobre el nuevo lanzamiento de producto de Tesla?", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Determina el tono emocional de este informe de análisis de mercado.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "¿Es este artículo de noticias financieras sobre Apple positivo o negativo?", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "Categoriza este titular de noticias empresariales en su sector correcto.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Clasifica este artículo financiero por tipo de industria.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "¿A qué categoría pertenece este informe económico?", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Clasifica esta actualización del mercado por sector financiero.", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "Explica este concepto en términos sencillos.", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "¿Cuál es la diferencia entre la memoria RAM y el almacenamiento?", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Resume los puntos principales de este artículo de investigación.", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "Ayúdame a entender por qué ocurre este error.", 'domain': 'general', 'task': 'question_answering'},

    # ---------------- FRENCH ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "Analyse le sentiment dans le dernier rapport trimestriel.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Quel est le sentiment concernant le nouveau lancement de produit de Tesla ?", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Détermine le ton émotionnel de ce rapport d'analyse de marché.", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "Cet article de presse financier sur Apple est-il positif ou négatif ?", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "Catégorise ce titre de l'actualité économique dans le bon secteur.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Classe cet article financier par type d'industrie.", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "À quelle catégorie appartient ce rapport économique ?", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "Classe cette mise à jour de marché par secteur financier.", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "Explique ce concept en termes simples.", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Quelle est la différence entre la RAM et le stockage ?", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "Résume les points principaux de cet article de recherche.", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "Aide-moi à comprendre pourquoi cette erreur se produit.", 'domain': 'general', 'task': 'question_answering'},

    # ---------------- JAPANESE ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "最新の四半期収益レポートの感情分析をしてください。", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "テスラの新商品発売に関する感情はどうですか？", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "この市場分析レポートの感情的なトーンを判断してください。", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "アップルに関するこの金融ニュース記事は肯定的ですか、否定的ですか？", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "このビジネスニュースの見出しを正しい業界に分類してください。", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "この金融記事を業界タイプで分類してください。", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "この経済レポートはどのカテゴリに属しますか？", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "この市場アップデートを金融セクター別に分類してください。", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "この概念をわかりやすく説明してください。", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "RAM とストレージの違いは何ですか？", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "この研究記事の主要な点を要約してください。", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "なぜこのエラーが発生するのか理解を手伝ってください。", 'domain': 'general', 'task': 'question_answering'},

    # ---------------- CHINESE (Simplified) ----------------
    # Finance Domain - Sentiment Analysis
    {'prompt': "分析最近季度财务报告的情绪。", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "关于特斯拉新产品发布的情绪如何？", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "确定这份市场分析报告的情感色调。", 'domain': 'finance', 'task': 'sentiment_analysis'},
    {'prompt': "这篇关于苹果的金融新闻文章是正面的还是负面的？", 'domain': 'finance', 'task': 'sentiment_analysis'},

    # Finance Domain - News Classification
    {'prompt': "将这条商业新闻标题归类到正确的行业。", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "按行业类型对这篇金融文章进行分类。", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "这份经济报告属于哪个类别？", 'domain': 'finance', 'task': 'news_classification'},
    {'prompt': "按金融部门对这次市场更新进行分类。", 'domain': 'finance', 'task': 'news_classification'},

    # General Domain
    {'prompt': "用简单的方式解释这个概念。", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "RAM 和存储有什么区别？", 'domain': 'general', 'task': 'question_answering'},
    {'prompt': "总结这篇研究文章的要点。", 'domain': 'general', 'task': 'text_summarization'},
    {'prompt': "帮我理解为什么会出现这个错误。", 'domain': 'general', 'task': 'question_answering'},
]
