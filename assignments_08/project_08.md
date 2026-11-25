## Scenario A — Lightweight compute: 
* A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, 
* running 8 hours per day, 5 days per week (approximately 160 hours per month). 
* Use the US East (N. Virginia) region.

Costs aproximately $1.66. I was surprised by how inexpensive this lightweight compute option is.

## Scenario B — Heavy analytics workload: 
* p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); 
* RDS db.m5.large instance (2 vCPU, 8 GB RAM); 
* S3 Standard storage bucket with 1 TB of data. 
* Use US East (N. Virginia).

Scenario B costs: 2,233.80 USD(EC2) + 323.03 USD (RDS) + 23.55 USD (S3) = 2,580.38 USD (Total) per month.
The difference between the two scenarios is significant: the heavy analytics workload costs over 1500 times more than the lightweight scenario.
This demostrates that GPU instances are expansive and are best suited for workloads that can take advantage of GPU acceleration, such as machine
learning training or large-scale scientific computign. For small data processing tasks or applications that do not require a GPU, a lightweight scenario is 
much more cost effective.

## Supabase setup
* python200 project is created
* credentials such as project url and key can be found on the Project Settings page
* SUPABASE_URL and SUPABASE_KEY has been added to the .env file into the Week 9 folder
* weather_raw and weather_enriched tables with disabled Row Level Security has been created
