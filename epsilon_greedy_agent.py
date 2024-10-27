import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class WikipediaCrawler:
    def __init__(self, base_url="https://en.wikipedia.org"):
        self.base_url = base_url
        self.visited = set()
    
    def crawl(self, start_subject, max_pages=10):
        pages_to_visit = [f"{self.base_url}/wiki/{start_subject}"]
        crawled_data = []

        while pages_to_visit and len(crawled_data) < max_pages:
            current_url = pages_to_visit.pop(0)
            if current_url in self.visited:
                continue
            
            self.visited.add(current_url)
            #print(f"Crawling: {current_url}")
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extracting text and links
            links = [urljoin(self.base_url, link['href']) for link in soup.find_all('a', href=True)]
            crawled_data.append((current_url, soup.get_text(), links))

            # Adding links to pages_to_visit
            for link in links:
                if link.startswith(self.base_url + "/wiki/") and link not in self.visited:
                    pages_to_visit.append(link)

        return crawled_data


class EpsilonGreedyAgent:
    def __init__(self, subject, epsilon=0.1, similarity_threshold=0.5):
        self.subject = subject
        self.epsilon = epsilon
        self.vectorizer = TfidfVectorizer()
        self.arm_rewards = np.zeros(2)
        self.arm_counts = np.zeros(2)
        self.similarity_threshold = similarity_threshold

    def select_action(self, uri):
        # Vectorize the subject and the URI
        all_texts = [self.subject, uri]
        vectors = self.vectorizer.fit_transform(all_texts).toarray()
        
        subject_vector = vectors[0].reshape(1, -1)
        uri_vector = vectors[1].reshape(1, -1)

        # Calculate similarity
        similarity = cosine_similarity(subject_vector, uri_vector)[0][0]
        
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:  # Exploration
            action = np.random.choice([0, 1])
        else:  # Exploitation
            action = 1 if self.arm_counts[1] > self.arm_counts[0] else 0

        return action, similarity


class EpsilonGreedyWebpageDownloader:
    def __init__(self, subject, download_dir, max_storage_mb=10, similarity_threshold=0.5, epsilon=0.1):
        self.crawler = WikipediaCrawler()
        self.agent = EpsilonGreedyAgent(subject, epsilon, similarity_threshold)
        self.subject = subject
        self.download_dir = download_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.similarity_threshold = similarity_threshold
        
        # Performance metrics
        self.total_reward = 0
        self.download_success_count = 0
        self.similarity_values = []
        self.rewards_over_time = []
        self.success_rate_over_time = []

        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)

    def get_storage_size(self):
        """Calculate the total storage size of the downloaded files."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.download_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def run(self, max_pages=10):
        crawled_data = self.crawler.crawl(self.subject, max_pages)

        for url, content, links in crawled_data:
            current_storage_size = self.get_storage_size()
            if current_storage_size >= self.max_storage_bytes:
                print(f"Maximum storage limit reached. Skipping further downloads.")
                break

            # Evaluate each URI separately
            for link in links:
                if link.startswith("https://en.wikipedia.org/wiki/"):
                    action, similarity = self.agent.select_action(link.split('/')[-1])

                    new_content_size = len(content.encode('utf-8'))

                    # Check for action: Download (1) or Skip (0)
                    if action == 1 and (current_storage_size + new_content_size <= self.max_storage_bytes) and similarity > self.similarity_threshold:
                        self.download_page(url, content, new_content_size)
                        self.download_success_count += 1
                        self.similarity_values.append(similarity)
                        reward = 1
                    else:
                        #print(f"Skipped: {url} (Similarity: {similarity})")
                        reward = 0

                    self.total_reward += reward
                    self.rewards_over_time.append(self.total_reward)
                    self.success_rate_over_time.append(self.download_success_count / (len(self.rewards_over_time)))
                    self.agent.arm_counts[action] += 1
                    self.agent.arm_rewards[action] += reward

    def download_page(self, url, content, content_size):
        filename = self.get_filename(url)
        file_path = os.path.join(self.download_dir, filename)
        #print(f"Downloaded: {url} (Size: {content_size} bytes)")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
    def get_filename(self, url):
        return url.split('/')[-1] + '.html'

    def evaluate_performance(self):
        average_similarity = np.mean(self.similarity_values) if self.similarity_values else 0
        storage_utilization = (self.get_storage_size() / self.max_storage_bytes) * 100
        
        print("\nPerformance Evaluation for subject '{}':".format(self.subject))
        print(f"Total Reward: {self.total_reward}")
        print(f"Download Success Rate: {self.download_success_count}")
        print(f"Average Similarity of Downloaded Pages: {average_similarity:.4f}")
        print(f"Storage Utilization: {storage_utilization:.2f}%")
        
        return self.rewards_over_time, self.success_rate_over_time, self.similarity_values

    def plot_performance(self, subject_results):
        plt.figure(figsize=(12, 8))

        for subject, results in subject_results.items():
            rewards_over_time, success_rate_over_time, similarity_values = results
            
            plt.subplot(3, 1, 1)
            plt.plot(rewards_over_time, label=f'Total Reward: {subject}')
        
            plt.subplot(3, 1, 2)
            plt.plot(success_rate_over_time, label=f'Success Rate: {subject}')
        
            average_similarity_over_time = [np.mean(similarity_values[:i+1]) for i in range(len(similarity_values))]
            plt.subplot(3, 1, 3)
            plt.plot(average_similarity_over_time, label=f'Average Similarity: {subject}')

        plt.subplot(3, 1, 1)
        plt.title('Total Reward Over Time')
        plt.xlabel('Pages Processed')
        plt.ylabel('Total Reward')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.title('Download Success Rate Over Time')
        plt.xlabel('Pages Processed')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.title('Average Similarity of Downloaded Pages Over Time')
        plt.xlabel('Pages Processed')
        plt.ylabel('Average Similarity')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    subjects = ["Artificial_intelligence", "Machine_learning", "Deep_learning"]
    download_directory = "downloads"
    subject_results = {subject: [] for subject in subjects}

    # Run the downloader for multiple epochs
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}")
        for subject in subjects:
            downloader = EpsilonGreedyWebpageDownloader(subject, download_directory, max_storage_mb=10, similarity_threshold=0.5, epsilon=0.1)
            downloader.run(max_pages=50)
            results = downloader.evaluate_performance()
            subject_results[subject].append(results)

    # Now we can plot all performance metrics
    all_rewards = {subject: [] for subject in subjects}
    all_success_rates = {subject: [] for subject in subjects}
    all_similarities = {subject: [] for subject in subjects}

    # Aggregate results from all epochs
    for subject in subjects:
        for epoch_results in subject_results[subject]:
            rewards_over_time, success_rate_over_time, similarity_values = epoch_results
            all_rewards[subject].extend(rewards_over_time)
            all_success_rates[subject].extend(success_rate_over_time)
            all_similarities[subject].extend(similarity_values)

    # Plot overall performance metrics
    plt.figure(figsize=(12, 8))

    for subject in subjects:
        # Total rewards over time
        plt.subplot(3, 1, 1)
        plt.plot(all_rewards[subject], label=f'Total Reward: {subject}')

        # Download success rate over time
        plt.subplot(3, 1, 2)
        plt.plot(all_success_rates[subject], label=f'Success Rate: {subject}')

        # Average similarity of downloaded pages over time
        average_similarity_over_time = [np.mean(all_similarities[subject][:i+1]) for i in range(len(all_similarities[subject]))]
        plt.subplot(3, 1, 3)
        plt.plot(average_similarity_over_time, label=f'Average Similarity: {subject}')

    plt.subplot(3, 1, 1)
    plt.title('Total Reward Over Time Across Epochs')
    plt.xlabel('Pages Processed')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title('Download Success Rate Over Time Across Epochs')
    plt.xlabel('Pages Processed')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title('Average Similarity of Downloaded Pages Over Time Across Epochs')
    plt.xlabel('Pages Processed')
    plt.ylabel('Average Similarity')
    plt.legend()

    plt.tight_layout()
    plt.show()
