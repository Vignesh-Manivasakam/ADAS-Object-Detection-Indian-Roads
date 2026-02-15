# UPDATED RESUME - PROJECT PORTFOLIO SECTION

## Add This to Your Resume's "Project Portfolio" Section

---

### R&D Digitalization & Automation

**AI-Driven Requirement Similarity Assistant (Internal Tool)**
• Tech Stack: Python, Azure OpenAI (GPT-4), LangChain, FAISS (Vector DB), Streamlit.
• Architecture: Designed a RAG (Retrieval Augmented Generation) pipeline to ingest PDF/Excel files, chunk requirements, and generate embeddings to identify redundant requirements across legacy projects.
• Key Feature: Built an "Exact-Match Filtering" layer before semantic analysis to reduce API token usage by 40%, optimizing cost-to-performance.
• Public Demo: Created a non-confidential Proof-of-Concept (AI Similarity Assist Tool) on GitHub to demonstrate the architecture.

**Agentic AI Workflow for System Requirement Review**
• Tech Stack: Python, Multi-Agent Systems, Prompt Engineering.
• Solution: Developed a multi-agent system with specialized personas: "Safety & Security Agent" (checking ISO 26262/SOTIF), "Test Verifiability Agent," and a "Data Layer" responsible for collecting and refining input specifications for the agents.
• Impact: Achieved a recall rate of >90% in identifying missing safety attributes compared to manual Domain Expert review.

**[NEW] ADAS Object Detection for Indian Road Scenarios (Hackathon Winner 🏆)**
• Tech Stack: Python, PyTorch, YOLOv8, Albumentations, INT8 Quantization, CUDA
• Architecture: Designed and deployed a YOLOv8-based multi-class detection system trained on 45,000+ images from Indian Driving Dataset (IDD), achieving production-grade performance across 13 object classes
• Progressive Training: Implemented 3-stage training pipeline (640px → 960px → 1280px) with adaptive hyperparameter tuning, improving baseline mAP by 47% (0.285 → 0.420)
• Data Engineering: Built comprehensive augmentation pipeline using Albumentations to simulate adverse conditions (rain, fog, night, glare), generating 20,000 synthetic images that improved low-light performance by 28%
• Optimization: Deployed INT8 dynamic quantization achieving 67% model size reduction (48MB → 16MB) and 2x inference speedup (12.8 → 24.5 FPS) while maintaining <2% accuracy drop for real-time edge deployment
• Results: **Won 1st Place in ADAS Hackathon** with 0.641 mAP@50, 0.773 Precision, 0.569 Recall | Open-sourced on GitHub with comprehensive documentation
• GitHub: github.com/Vignesh-Manivasakam/ADAS-Object-Detection-Indian-Roads

**Digital Test Lab Management System**
• Role: Project Lead & Process Architect (Managed External Development Team).
• Scope: Defined the comprehensive business logic and system architecture to digitize the end-to-end testing workflow (Request -> Scheduling -> Report Generation).
• Result: Established standardized process flowcharts, enabling real-time equipment utilization tracking and reducing administrative delays.

**PDF to RTF Conversion for DOORS Import**
• Tech Stack: Python, PyPDF2, ReportLab, Regular Expressions
• Challenge: Engineering teams received customer requirements as PDFs that couldn't be directly imported into IBM DOORS requirements management tool
• Solution: Developed automated conversion pipeline that extracts text from PDFs, preserves formatting/structure, and generates DOORS-compatible RTF files
• Impact: Reduced manual requirement entry time by 85% (from 2 days to 2 hours per document), eliminated transcription errors
• GitHub: [Link to sanitized demo version]

**AI-Based Technical Review System for Requirements**
• Tech Stack: Python, OpenAI API, NLP, Regex, Custom Rule Engine
• Architecture: Built multi-stage review system that validates requirements against ASPICE SWE.1 criteria including unambiguity, consistency, traceability, and verifiability
• Key Features:
  - Grammar/spell checking using LanguageTool API
  - Passive voice detection and recommendations
  - Weak word identification ("should", "may", "etc.")
  - Cross-reference validation (checks if linked requirements exist)
  - Completeness checking (ensures all mandatory attributes present)
• Impact: Reduced initial review cycle time by 40%, identified 300+ issues before human review across 5 pilot projects
• GitHub: [Link to demo with synthetic requirements]

---

### Systems Engineering & Core Product Development

**Standardization of Non-functional requirements**
• Focus: Platform Consistency & Reusability.
• Scope: Consolidated NFRs across 10+ customer projects, creating a unified library for Performance, Durability, and Environmental standards to eliminate redundancy.
• Impact: Achieved a 40% reduction in engineering effort for new project acquisition by enabling modular reuse of requirement baselines across OEMs.

**Electrohydraulic Power Steering Pumps**
• Domain: Commercial Vehicle Steering | Tools: VFD, MCT Software, RLDA.
• Scope: Defined test bench specifications based on Road Load Data Analysis (RLDA). Developed a VBA calculation tool for motor selection to optimize the weight-to-power ratio.
• Validation: Conducted extensive motor and vehicle-level testing, validating performance variations and ensuring system efficiency met OEM targets.

**Variable Displacement Pump**
• Domain: Hydraulic Systems | Focus: NVH & Performance.
• Impact: Led the development from concept to prototype, conducting 100+ test trials to resolve noise issues (NVH). Achieved a 0.6% improvement in fuel efficiency during vehicle testing.

---

## Updated Technical Skills Section

### Technical Skills (Add These)

**AI & Machine Learning**:
- Computer Vision: YOLOv8, Object Detection, Image Classification, Data Augmentation (Albumentations)
- Model Optimization: INT8 Quantization, ONNX Export, TensorRT, Edge Deployment
- Deep Learning Frameworks: PyTorch, TensorFlow, Ultralytics

**Programming & Tools**:
- Python: NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn
- Version Control: Git, GitHub, Git LFS
- ML Ops: Weights & Biases, TensorBoard, Model Versioning

**Deployment & Hardware**:
- Edge AI: NVIDIA Jetson, Raspberry Pi, INT8/FP16 Quantization
- Performance Optimization: CUDA, GPU Acceleration, Inference Optimization

---

## LinkedIn About Section Update

Add this paragraph to your LinkedIn "About" section:

```
Beyond my core expertise in automotive systems engineering, I'm passionate about 
leveraging AI/ML to solve real-world problems. Recently won 1st place in an ADAS 
Hackathon by building a YOLOv8-based object detection system optimized for Indian 
road scenarios. The project involved progressive training strategies, advanced data 
augmentation for adverse weather conditions, and INT8 quantization for edge deployment
— achieving 0.641 mAP@50 and real-time performance on embedded hardware. This 
experience reinforced my belief that the future of automotive engineering lies at the 
intersection of traditional domain expertise and modern AI capabilities.

Open-source code and detailed documentation: 
github.com/Vignesh-Manivasakam/ADAS-Object-Detection-Indian-Roads
```

---

## Resume Bullet Point Priority

**Use This Order** (most impressive first):

1. **ADAS Hackathon Winner** (shows competition success + technical depth)
2. **AI-Driven Requirement Similarity Assistant** (shows internal impact at Bosch)
3. **Agentic AI Workflow for System Requirement Review** (shows advanced AI concepts)
4. **AI-Based Technical Review System** (shows practical automation)
5. **PDF to RTF Conversion** (shows problem-solving for pain points)
6. **Digital Test Lab Management System** (shows project leadership)

---

## GitHub Profile README Update

Create `github.com/Vignesh-Manivasakam/README.md`:

```markdown
# Hi, I'm Vignesh Manivasakam 👋

## 🚗 Automotive Systems Engineer | AI/ML Enthusiast

I bridge traditional engineering (ASPICE, ISO 26262) with modern AI to digitalize 
R&D workflows and build intelligent automotive systems.

### 🏆 Recent Achievements
- 🥇 **Won ADAS Hackathon** with YOLOv8-based object detection system
- ⚡ **Reduced engineering lead time by 65%** through GenAI-powered requirement reviews
- 🎯 **CPRE-FL Certified** in Requirements Engineering

### 🔧 Tech Stack
`Python` `PyTorch` `YOLOv8` `Azure OpenAI` `RAG` `Agentic AI` `Computer Vision`

### 📂 Featured Projects
- 🚗 [ADAS Object Detection](https://github.com/Vignesh-Manivasakam/ADAS-Object-Detection-Indian-Roads) - Hackathon-winning detection system
- 🤖 [AI Similarity Assist](https://github.com/Vignesh-Manivasakam/AI-Similarity-Assist) - RAG-based requirement deduplication
- 📄 [Requirements Reviewer](https://github.com/Vignesh-Manivasakam/AI-Requirements-Reviewer) - Automated ASPICE compliance checker

### 📫 Let's Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/vignesh-manivasakam)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:vicky.manivasagam@gmail.com)

![GitHub stats](https://github-readme-stats.vercel.app/api?username=Vignesh-Manivasakam&show_icons=true&theme=radical)
```

---

## Interview Preparation

**Expected Question**: "I see you have an ADAS project on your resume. Tell me about it."

**Your Answer (45 seconds)**:
"I won first place in an ADAS Hackathon by building an object detection system specifically optimized for Indian road scenarios. The challenge was that Indian roads have unique characteristics — dense mixed traffic with auto-rickshaws, poor lighting conditions, and diverse infrastructure — that standard ADAS models trained on Western datasets struggle with.

I used YOLOv8 and implemented a three-stage progressive training strategy, starting at 640-pixel resolution for fast convergence, then fine-tuning at 960 and 1280 pixels to capture small objects like distant traffic lights. I also built a comprehensive augmentation pipeline simulating rain, fog, and night conditions, which generated 20,000 synthetic training images.

The final model achieved 0.641 mAP@50 with 77% precision. To make it production-ready, I applied INT8 quantization, reducing the model size by 67% and doubling inference speed to 24 FPS — suitable for edge devices like NVIDIA Jetson.

I open-sourced the entire project on GitHub with full documentation, and it's gotten good traction from the computer vision community."

**Follow-up Questions You Should Be Ready For**:
1. "What was the most challenging part?"
2. "How did you validate the model?"
3. "Why YOLOv8 instead of other detectors?"
4. "How would you improve this further?"

(See RESUME_INTEGRATION.md for detailed answers)

---

## Final Checklist Before Updating Resume

- [ ] GitHub repository is public and polished
- [ ] README has professional formatting with badges
- [ ] All code is well-commented
- [ ] Demo GIF or images are included
- [ ] LICENSE file is present
- [ ] GitHub profile README is updated
- [ ] LinkedIn profile mentions the project
- [ ] You can explain every technical decision (be ready for deep dives)
- [ ] You have practiced the 45-second elevator pitch
- [ ] Resume PDF is updated and tested (check links work)

**Once complete**: Update your resume, push to GitHub, post on LinkedIn, and start applying! 🚀
