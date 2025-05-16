define({ entries : {
    "Brown2020": {
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.",
        "author": "Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei",
        "keywords": "Language Models, Few-Shot Learning, GPT-3, Transformer, Natural Language Processing, Pre-training, Zero-Shot Learning, One-Shot Learning, Deep Learning, Text Generation",
        "month": "5",
        "title": "Language Models are Few-Shot Learners",
        "type": "article",
        "url": "http://arxiv.org/abs/2005.14165",
        "year": "2020"
    },
    "Deng2024": {
        "abstract": "Large Language Models (LLMs) have achieved unparalleled success across diverse language modeling tasks in recent years. However, this progress has also intensified ethical concerns, impacting the deployment of LLMs in everyday contexts. This paper provides a comprehensive survey of ethical challenges associated with LLMs, from longstanding issues such as copyright infringement, systematic bias, and data privacy, to emerging problems like truthfulness and social norms. We critically analyze existing research aimed at understanding, examining, and mitigating these ethical risks. Our survey underscores integrating ethical standards and societal values into the development of LLMs, thereby guiding the development of responsible and ethically aligned language models.",
        "author": "Chengyuan Deng and Yiqun Duan and Xin Jin and Heng Chang and Yijun Tian and Han Liu and Yichen Wang and Kuofeng Gao and Henry Peng Zou and Yiqiao Jin and Yijia Xiao and Shenghao Wu and Zongxing Xie and Weimin Lyu and Sihong He and Lu Cheng and Haohan Wang and Jun Zhuang",
        "keywords": "Large Language Models, Ethics, Bias, Privacy, Fairness, Transparency, Accountability, Social Impact, AI Governance, Emerging Dilemmas",
        "month": "6",
        "title": "Deconstructing The Ethics of Large Language Models from Long-standing Issues to New-emerging Dilemmas: A Survey",
        "type": "article",
        "url": "http://arxiv.org/abs/2406.05392",
        "year": "2024"
    },
    "Devlin2018": {
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).",
        "author": "Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova",
        "keywords": "BERT, Bidirectional Transformer, Pre-training, Language Understanding, Natural Language Processing, Deep Learning, Masked Language Modeling, Transfer Learning, Contextual Embeddings",
        "month": "10",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "type": "article",
        "url": "http://arxiv.org/abs/1810.04805",
        "year": "2018"
    },
    "Hadi2023": {
        "abstract": "<p>&lt;p&gt;Within the vast expanse of computerized language processing, a revolutionary entity known as Large Language Models (LLMs) has emerged, wielding immense power in its capacity to comprehend intricate linguistic patterns and conjure coherent and contextually fitting responses. Large language models (LLMs) are a type of artificial intelligence (AI) that have emerged as powerful tools for a wide range of tasks, including natural language processing (NLP), machine translation, and question-answering. This survey paper provides a comprehensive overview of LLMs, including their history, architecture, training methods, applications, and challenges. The paper begins by discussing the fundamental concepts of generative AI and the architecture of generative pre- trained transformers (GPT). It then provides an overview of the history of LLMs, their evolution over time, and the different training methods that have been used to train them. The paper then discusses the wide range of applications of LLMs, including medical, education, finance, and engineering. It also discusses how LLMs are shaping the future of AI and how they can be used to solve real-world problems. The paper then discusses the challenges associated with deploying LLMs in real-world scenarios, including ethical considerations, model biases, interpretability, and computational resource requirements. It also highlights techniques for enhancing the robustness and controllability of LLMs, and addressing bias, fairness, and generation quality issues. Finally, the paper concludes by highlighting the future of LLM research and the challenges that need to be addressed in order to make LLMs more reliable and useful. This survey paper is intended to provide researchers, practitioners, and enthusiasts with a comprehensive understanding of LLMs, their evolution, applications, and challenges. By consolidating the state-of-the-art knowledge in the field, this survey serves as a valuable resource for further advancements in the development and utilization of LLMs for a wide range of real-world applications. The GitHub repo for this project is available at https://github.com/anas-zafar/LLM-Survey&lt;/p&gt;</p>",
        "author": "Muhammad Usman Hadi and qasem al tashi and Rizwan Qureshi and Abbas Shah and amgad muneer and Muhammad Irfan and Anas Zafar and Muhammad Bilal Shaikh and Naveed Akhtar and Jia Wu and Seyedali Mirjalili",
        "doi": "10.36227/techrxiv.23589741",
        "institution": "TechRxiv",
        "keywords": "Large Language Models, Natural Language Processing, Applications, Challenges, Limitations, Future Prospects, Deep Learning, Transformer, Pre-trained Models, Artificial Intelligence",
        "month": "11",
        "title": "Large Language Models: A Comprehensive Survey of its Applications, Challenges, Limitations, and Future Prospects",
        "type": "misc",
        "url": "https://www.techrxiv.org/articles/preprint/A_Survey_on_Large_Language_Models_Applications_Challenges_Limitations_and_Practical_Usage/23589741",
        "year": "2023"
    },
    "Hajikhani2024": {
        "abstract": "This paper examines the comparative effectiveness of a specialized compiled language model and a general-purpose model such as OpenAI\u2019s GPT-3.5 in detecting sustainable development goals (SDGs) within text data. It presents a critical review of large language models (LLMs), addressing challenges related to bias and sensitivity. The necessity of specialized training for precise, unbiased analysis is underlined. A case study using a company descriptions data set offers insight into the differences between the GPT-3.5 model and the specialized SDG detection model. While GPT-3.5 boasts broader coverage, it may identify SDGs with limited relevance to the companies\u2019 activities. In contrast, the specialized model zeroes in on highly pertinent SDGs. The importance of thoughtful model selection is emphasized, taking into account task requirements, cost, complexity, and transparency. Despite the versatility of LLMs, the use of specialized models is suggested for tasks demanding precision and accuracy. The study concludes by encouraging further research to find a balance between the capabilities of LLMs and the need for domain-specific expertise and interpretability.",
        "author": "Arash Hajikhani and Carolyn Cole",
        "doi": "10.1162/qss_a_00310",
        "issn": "26413337",
        "journal": "Quantitative Science Studies",
        "keywords": "Large Language Models, Sensitivity, Bias, Specialized AI, Fairness, Robustness, Natural Language Processing, Machine Learning, Model Evaluation, Ethics",
        "month": "8",
        "pages": "1-21",
        "publisher": "MIT Press",
        "title": "A critical review of large language models: Sensitivity, bias, and the path toward specialized AI",
        "type": "article",
        "year": "2024"
    },
    "Hou2023": {
        "abstract": "Large Language Models (LLMs) have significantly impacted numerous domains, including Software Engineering (SE). Many recent publications have explored LLMs applied to various SE tasks. Nevertheless, a comprehensive understanding of the application, effects, and possible limitations of LLMs on SE is still in its early stages. To bridge this gap, we conducted a systematic literature review (SLR) on LLM4SE, with a particular focus on understanding how LLMs can be exploited to optimize processes and outcomes. We select and analyze 395 research papers from January 2017 to January 2024 to answer four key research questions (RQs). In RQ1, we categorize different LLMs that have been employed in SE tasks, characterizing their distinctive features and uses. In RQ2, we analyze the methods used in data collection, preprocessing, and application, highlighting the role of well-curated datasets for successful LLM for SE implementation. RQ3 investigates the strategies employed to optimize and evaluate the performance of LLMs in SE. Finally, RQ4 examines the specific SE tasks where LLMs have shown success to date, illustrating their practical contributions to the field. From the answers to these RQs, we discuss the current state-of-the-art and trends, identifying gaps in existing research, and flagging promising areas for future study. Our artifacts are publicly available at https://github.com/xinyi-hou/LLM4SE_SLR.",
        "author": "Xinyi Hou and Yanjie Zhao and Yue Liu and Zhou Yang and Kailong Wang and Li Li and Xiapu Luo and David Lo and John Grundy and Haoyu Wang",
        "keywords": "Large Language Models, Software Engineering, Systematic Literature Review, Code Generation, Natural Language Processing, Machine Learning, Program Analysis, Automated Software Development, Deep Learning, AI-assisted Programming",
        "month": "8",
        "title": "Large Language Models for Software Engineering: A Systematic Literature Review",
        "type": "article",
        "url": "http://arxiv.org/abs/2308.10620",
        "year": "2023"
    },
    "Naveed2023": {
        "abstract": "Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language processing tasks and beyond. This success of LLMs has led to a large influx of research contributions in this direction. These works encompass diverse topics such as architectural innovations, better training strategies, context length improvements, fine-tuning, multi-modal LLMs, robotics, datasets, benchmarking, efficiency, and more. With the rapid development of techniques and regular breakthroughs in LLM research, it has become considerably challenging to perceive the bigger picture of the advances in this direction. Considering the rapidly emerging plethora of literature on LLMs, it is imperative that the research community is able to benefit from a concise yet comprehensive overview of the recent developments in this field. This article provides an overview of the existing literature on a broad range of LLM-related concepts. Our self-contained comprehensive overview of LLMs discusses relevant background concepts along with covering the advanced topics at the frontier of research in LLMs. This review article is intended to not only provide a systematic survey but also a quick comprehensive reference for the researchers and practitioners to draw insights from extensive informative summaries of the existing works to advance the LLM research.",
        "author": "Humza Naveed and Asad Ullah Khan and Shi Qiu and Muhammad Saqib and Saeed Anwar and Muhammad Usman and Naveed Akhtar and Nick Barnes and Ajmal Mian",
        "keywords": "Large Language Models, Transformer, Natural Language Processing, Deep Learning, Pre-training, Fine-tuning, Multimodal Models, Model Scaling, Language Understanding, AI Applications",
        "month": "7",
        "title": "A Comprehensive Overview of Large Language Models",
        "type": "article",
        "url": "http://arxiv.org/abs/2307.06435",
        "year": "2023"
    },
    "Patil2024": {
        "abstract": "Natural language processing (NLP) has significantly transformed in the last decade, especially in the field of language modeling. Large language models (LLMs) have achieved SOTA performances on natural language understanding (NLU) and natural language generation (NLG) tasks by learning language representation in self-supervised ways. This paper provides a comprehensive survey to capture the progression of advances in language models. In this paper, we examine the different aspects of language models, which started with a few million parameters but have reached the size of a trillion in a very short time. We also look at how these LLMs transitioned from task-specific to task-independent to task-and-language-independent architectures. This paper extensively discusses different pretraining objectives, benchmarks, and transfer learning methods used in LLMs. It also examines different finetuning and in-context learning techniques used in downstream tasks. Moreover, it explores how LLMs can perform well across many domains and datasets if sufficiently trained on a large and diverse dataset. Next, it discusses how, over time, the availability of cheap computational power and large datasets have improved LLM\u2019s capabilities and raised new challenges. As part of our study, we also inspect LLMs from the perspective of scalability to see how their performance is affected by the model\u2019s depth, width, and data size. Lastly, we provide an empirical comparison of existing trends and techniques and a comprehensive analysis of where the field of LLM currently stands.",
        "author": "Rajvardhan Patil and Venkat Gudivada",
        "doi": "10.3390/app14052074",
        "issn": "20763417",
        "issue": "5",
        "journal": "Applied Sciences (Switzerland)",
        "keywords": "LLMs,NLP,PLMs,language models,large language model,literature review,natural language processing,review,survey",
        "month": "3",
        "publisher": "Multidisciplinary Digital Publishing Institute (MDPI)",
        "title": "A Review of Current Trends, Techniques, and Challenges in Large Language Models (LLMs)",
        "type": "misc",
        "volume": "14",
        "year": "2024"
    },
    "Vaswani2017": {
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
        "author": "Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin",
        "keywords": "Transformer, Self-Attention, Neural Networks, Sequence Modeling, Deep Learning, Machine Translation, Natural Language Processing, Attention Mechanism, Encoder-Decoder Architecture",
        "month": "6",
        "title": "Attention Is All You Need",
        "type": "article",
        "url": "http://arxiv.org/abs/1706.03762",
        "year": "2017"
    },
    "Zhao2023": {
        "abstract": "Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable AI algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pre-training Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. Since researchers have found that model scaling can lead to performance improvement, they further study the scaling effect by increasing the model size to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement but also show some special abilities that are not present in small-scale language models. To discriminate the difference in parameter scale, the research community has coined the term large language models (LLM) for the PLMs of significant size. Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT, which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. In this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Besides, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions.",
        "author": "Wayne Xin Zhao and Kun Zhou and Junyi Li and Tianyi Tang and Xiaolei Wang and Yupeng Hou and Yingqian Min and Beichen Zhang and Junjie Zhang and Zican Dong and Yifan Du and Chen Yang and Yushuo Chen and Zhipeng Chen and Jinhao Jiang and Ruiyang Ren and Yifan Li and Xinyu Tang and Zikang Liu and Peiyu Liu and Jian-Yun Nie and Ji-Rong Wen",
        "keywords": "Large Language Models, Survey, Transformer, Pre-training, Natural Language Processing, Deep Learning, Model Architecture, Fine-tuning, Language Generation, AI Trends",
        "month": "3",
        "title": "A Survey of Large Language Models",
        "type": "article",
        "url": "http://arxiv.org/abs/2303.18223",
        "year": "2023"
    }
}});