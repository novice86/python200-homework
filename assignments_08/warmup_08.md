## Cloud Concepts Question 1
The core model is pay-as-you-go: computing is a utility, paid for as an operating expense (OpEx) only for what you actually use. Owning servers is the opposite —  a large upfront capital expense (CapEx), plus power, cooling, and maintenance, whether they're busy or idle, and capacity bought for peak demand in advance. In the cloud, cost follows demand, so you can scale up and down and stop paying for resources you release, while managed services let you build end-to-end without running each piece yourself. The trade-off is that at sustained, predictable, high utilisation, owning servers can be cheaper per unit, and cloud costs can grow quickly if resources are left running unused.

## Cloud Concepts Question 2
### Difference

**Vertical scaling (scale up)**  means making one machine bigger — more CPU, RAM, or a faster GPU. It's simple, because nothing changes about how the system is distributed, but it has a hard ceiling (there's a biggest machine) and that one machine is a single point of failure, and resizing it often needs downtime.

**Horizontal scaling (scale out)** means adding more machines and spreading the work across them. It has no single-machine ceiling and is more fault-tolerant, but it needs the work to be splittable and something to distribute it — a load balancer, a queue, or partitioning.

**Concrete examples:** I'd choose vertical scaling to upgrade a database server from 16 GB RAM to 128 GB, or to move a model-training job to a machine with a faster GPU and more RAM. I'd choose horizontal scaling to add web servers behind a load balancer when traffic grows.
Scenarios

---


A. Horizontal scaling — the traffic is 100x higher, so add many identical web servers behind a load balancer to share the requests.

B. Vertical scaling — the single training job needs a more powerful machine, so move it to one with a faster GPU and more RAM.

C. Horizontal scaling — the files are independent, so split the 10,000 files across many machines and process them in parallel.

## Cloud Concepts Question 3

### Service Classifications

* **Gmail — SaaS:** It is a finished, ready-to-use email application delivered over the web where users manage messages and settings without maintaining servers or code.
* **Azure Virtual Machines — IaaS:** It provides on-demand virtualized compute instances (CPU, memory, networking, and OS choice) while leaving the OS configuration, runtime, and application stack up to engineer.
* **AWS S3 (Simple Storage Service) — IaaS:** It supplies raw, scalable object storage infrastructure accessed over APIs without requiring user to provision or maintain storage disks or file servers.
* **GitHub Codespaces — PaaS:** It provisions and manages preconfigured cloud development environments and runtimes automatically so developers can write and run code right away without managing underlying host infrastructure.
* **Snowflake — PaaS:** It abstracts away cluster provisioning, storage layout, and compute scaling so engineer can run SQL queries and data pipelines without administering database servers.
* **Supabase — BaaS:** It provides prebuilt backend infrastructure capabilities—such as a managed Postgres database, authentication, instant APIs, and storage—so frontend/client apps do not have to build backend plumbing from scratch.

---


### Service Model Definitions & Developer Responsibilities

#### IaaS (Infrastructure as a Service)
IaaS rents low-level building blocks—virtual machines, block/object storage, and virtual networks—while the cloud provider handles the physical data centers, hardware, power, cooling, and virtualization layer.
* **Example:** Azure Virtual Machines
* **Developer Responsibilities:** Configuring and patching the operating system, securing network/firewall rules, installing runtimes and dependencies, deploying and scaling the application, and managing backups or monitoring on top of the VM.

#### PaaS (Platform as a Service)
PaaS gives you a ready-made execution or runtime environment where the provider automatically provisions, patches, and scales the underlying servers, OS, and middleware so you can focus on your code or queries.
* **Example:** GitHub, Databricks
* **Developer Responsibilities:** Writing and maintaining application code, managing configuration/environment variables and dependencies, and handling data or queries, while letting the platform manage container/VM lifecycle, runtime maintenance, and scaling mechanics.

#### SaaS (Software as a Service)
SaaS delivers complete, end-user software applications over the internet on a subscription or usage basis, with the vendor running and maintaining the entire stack from hardware to user interface.
* **Example:** Gmail
* **Developer Responsibilities:** You do not manage infrastructure, code, or deployments—your responsibility is limited to configuring account/workspace settings, access permissions, or building light integrations via APIs or webhooks if needed.

## Cloud Concepts Question 4

### What is a Managed Data Platform?

A managed data platform (like **Databricks** or **Snowflake**) is an all-in-one cloud workspace built for data work. It sits on top of cloud providers like AWS, Azure, or GCP. It bundles storage, compute engines, security, and SQL/Python tools into one place so you can process data without managing servers.

### How Does It Differ from Using AWS or GCP Directly?

* **AWS / GCP Directly:** You get raw building blocks (storage buckets, virtual machines, networks, and databases). You have to connect, configure, and maintain all the pieces yourself.
* **Managed Data Platform:** The platform connects those pieces for you behind the scenes. You log into one interface, write your SQL or Python code, and let the platform run and scale the clusters automatically.

---

### What You Gain

* **Faster Setup:** You can start querying data and building pipelines right away instead of spending weeks configuring servers and networks.
* **Less Maintenance:** The platform handles server updates, cluster scaling, and performance tuning for you.
* **Unified Tools:** Data engineers, analysts, and data scientists can work together in one workspace with shared security and governance.
* **Pay-Per-Query / Auto-Pause:** Compute clusters can turn on when a job runs and shut off automatically when idle, so you do not pay for unused servers.

---

### What You Give Up

* **Low-Level Control:** You cannot customize the underlying operating system, hardware settings, or custom network rules as deeply as you can on AWS or GCP.
* **Extra Platform Cost:** You pay a platform fee (such as Snowflake credits or Databricks DBUs) on top of the underlying cloud storage and compute costs.
* **Flexibility for Unusual Workloads:** If you need custom background software or non-standard data tools, a managed platform can feel restrictive.

## Cloud Concepts Question 5

### The lesson describes a situation where the cloud is probably not the right choice. What is that situation?
Whether to use a solution provided by the cloud ecosystem really depends on the task requirements. 
If you are just building a prototype for a dataset that fits comfortably on a single machine and there are no massive compute demands, 
local processing is usually faster and much cheaper.

## Cloud Landscape Question 1

### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

* **Amazon Web Services (AWS):** AWS stands out for its broad service catalog, and global infrastructure footprint, making it a go-to choice for startups, high-growth digital businesses, and large enterprises building diverse cloud workloads.
* **Microsoft Azure:** Azure excels at seamless integration with Microsoft Active Directory and hybrid/on-premises environments, making it especially attractive to large enterprises and regulated industries already invested in Windows, Active Directory, and Microsoft 365.
* **Google Cloud Platform (GCP):** GCP is particularly strong in data analytics, cloud-native container orchestration, and AI/ML tooling, appealing to data-intensive organizations, AI-focused teams.

## Cloud Landscape Question 2

### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

* **Access** Azure requires organizational provisioning—joining a tenant, waiting on invitations, and configuring authentication—which can block late joiners or users hitting tenant issues for days.
* **Pedagogical fit** Supabase stores data as rows and columns in a relational database. Skills that students aquire are transfarable and more suited for the data role.
* **Pipeline coherence** The ETL pipeline we build in weeks 9–11 uses a raw zone and an enriched zone modeled as two related tables in Supabase, reinforcing this week's data modeling concepts.

## Cloud Landscape Question 3

### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

* You need to store 10 TB of image files and retrieve them by filename from any machine. - Object storage (AWS S3, GCP Cloud Storage, Azure Blob Storage)
* You need to run an ML training job on a GPU for four hours, then shut it down. - ML platform (AWS SageMaker, GCP Vertex AI, Azure ML)
* You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets. - Compute (AWS EC2, GCP Compute Engine, Azure VM)
* You need to send structured data to a large language model and get a text response back. - LLM API (AWS Bedrock, GCP Vertex AI, Azure Open AI)

## Cloud Landscape Question 4

### Sample Data Project & Stack

I would build a **City Climate & Air Quality Tracker** that collects daily temperature, precipitation, and air-quality readings for several metro areas, cleans and normalizes data, and stores raw vs. cleaned records for dashboarding.

* **Ingestion & Raw Storage:** A Python script (run locally or in **GitHub Actions** as PaaS) fetches public API/web data and drops raw JSON/CSV snapshots into **AWS S3** (IaaS/object storage).
* **Transformation & Enriched Storage:** A pipeline validates and enriches the records, loading raw and curated tables into **Supabase** (BaaS / managed Postgres) or **Snowflake** (PaaS data platform) for querying and app/dashboard access.

### Benefits of Consolidating to One Provider

* **Simpler Operations & Security:** You manage one identity/permissions model, one billing account, and fewer cross-service network connections or credentials.
* **Lower Data Movement Friction:** Keeping storage, compute, and querying in one ecosystem avoids cross-provider network latency and data egress fees.
* **Faster Development & Debugging:** Fewer moving parts and unified monitoring/logs make pipelines easier to set up and troubleshoot.

### What You Give Up by Consolidating

* **Best-of-Breed Tooling:** A single provider may not have the strongest tool for every stage of the pipeline (for example, pairing specialized storage or analytics tools across ecosystems).
* **Portability & Lock-In Protection:** Relying on one vendor's proprietary features, SQL extensions, or workflows makes switching or negotiating harder later.
* **Redundancy Across Providers:** Concentrating everything in one provider increases exposure if that provider experiences a major outage or regional issue.