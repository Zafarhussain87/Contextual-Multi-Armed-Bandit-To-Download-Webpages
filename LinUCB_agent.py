import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class WikipediaCrawler:
    def __init__(self, base_url="https://en.wikipedia.org"):
        self.base_url = base_url
        self.visited = set()

    def crawl(self, start_subject, max_pages=10):
        urls_to_visit = [f"{self.base_url}/wiki/{start_subject}"]
        crawled_data = []

        while urls_to_visit and len(crawled_data) < max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in self.visited:
                continue
            
            self.visited.add(current_url)
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, "html.parser")

            links = [urljoin(self.base_url, link['href']) for link in soup.find_all('a', href=True)]
            crawled_data.append((current_url, soup.get_text(), links))

            for link in links:
                if link.startswith(self.base_url + "/wiki/") and link not in self.visited:
                    urls_to_visit.append(link)

        return crawled_data


class LinUCB:
    def __init__(self, d, alpha=1.0):
        self.d = d
        self.alpha = alpha
        self.A = np.identity(d)
        self.b = np.zeros(d)

    def choose_arm(self, context_features):
        context_features = context_features.flatten()  # Ensure context is 1D
        assert context_features.shape[0] == self.d, f"Expected {self.d}, got {context_features.shape[0]}"

        theta = np.linalg.inv(self.A) @ self.b

        # p should be calculated with both arrays being 1D
        p = np.dot(theta, context_features) + self.alpha * np.sqrt(np.dot(context_features.T, np.linalg.inv(self.A)) @ context_features)
        return 1 if p > 0 else 0  # 1 for download, 0 for don't download

    def update(self, context_features, action, reward):
        context_features = context_features.flatten()  # Ensure it's a 1D array
        assert context_features.shape[0] == self.d, f"Expected {self.d}, got {context_features.shape[0]}"
        
        self.A += np.outer(context_features, context_features)
        self.b += reward * context_features


class LinUCBWebpageDownloader:
    def __init__(self, subject, download_dir, max_storage_mb=10, similarity_threshold=0.5):
        self.crawler = WikipediaCrawler()
        self.subject = subject
        self.download_dir = download_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.similarity_threshold = similarity_threshold
        os.makedirs(self.download_dir, exist_ok=True)
        self.total_reward = 0
        self.download_success_count = 0
        self.vectorizer = TfidfVectorizer()  # Do not set max_features yet
        
        self.total_reward = 0
        self.download_success_count = 0
        self.similarity_values = []
        self.rewards_over_time = []
        self.success_rate_over_time = []

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
        
        # Fit the vectorizer on both the subject and crawled URIs to determine n_features
        self.vectorizer.fit([self.subject] + [url.split('/')[-1] for url, _, _ in crawled_data])
        n_features = self.vectorizer.transform([self.subject]).shape[1]

        # Initialize LinUCB with the dynamic number of features
        self.agent = LinUCB(d=n_features)

        subject_vector = self.vectorizer.transform([self.subject]).toarray()[0]

        for url, content, links in crawled_data:
            current_storage_size = self.get_storage_size()
            if current_storage_size >= self.max_storage_bytes:
                print("Maximum storage limit reached. Skipping further downloads.")
                break

            uri = url.split('/')[-1]
            uri_vector = self.vectorizer.transform([uri]).toarray()[0]

            # Ensure uri_vector is 1D
            uri_vector = uri_vector.flatten()

            similarity = cosine_similarity(subject_vector.reshape(1, -1), uri_vector.reshape(1, -1))[0][0]

            action = self.agent.choose_arm(uri_vector)  # Pass the 1D array directly

            if action == 1 and (current_storage_size + len(content.encode('utf-8')) <= self.max_storage_bytes) and similarity >= self.similarity_threshold:
                self.download_page(url, content)
                reward = 1
                self.download_success_count += 1
                self.similarity_values.append(similarity)
            else:
                reward = 0

            self.total_reward += reward
            self.rewards_over_time.append(self.total_reward)
            self.success_rate_over_time.append(self.download_success_count / (len(self.rewards_over_time)))
            self.total_reward += reward
            self.agent.update(uri_vector, action, reward)  # Update with 1D context
            

    def download_page(self, url, content):
        filename = self.get_filename(url)
        file_path = os.path.join(self.download_dir, filename)
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


if __name__ == "__main__":
    subjects = ["Artificial_intelligence", "Machine_learning", "Deep_learning"]
    download_directory = "downloads"
    subject_results = {subject: [] for subject in subjects}

    # Run the downloader for multiple epochs
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}")
        for subject in subjects:
            downloader = LinUCBWebpageDownloader(subject, download_directory, max_storage_mb=10, similarity_threshold=0.2)
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
