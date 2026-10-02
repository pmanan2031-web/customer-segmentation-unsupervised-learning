# Customer Segmentation --- Unsupervised Learning

## 1. Business Problem and Dataset

The objective of this project is to segment customers based on their
purchasing behaviour so that businesses can design more targeted
marketing strategies. The Online Retail dataset was used. To reduce
noise, only customers from the United Kingdom were considered. Rows with
missing CustomerID, invalid quantities, and non-positive prices were
removed.

## 2. RFM Feature Engineering and Preprocessing

RFM analysis was used to describe customer behaviour through three
features:

-   Recency: Number of days since the customer's most recent purchase.
-   Frequency: Number of unique invoices/orders placed by the customer.
-   Monetary: Total amount spent by the customer.

TotalPrice was created using Quantity × UnitPrice. Frequency and
Monetary were log-transformed because they were highly right-skewed. The
RFM features were standardized using StandardScaler before clustering.
Outliers were handled using IQR-based capping.

## 3. Clustering Algorithms

Three unsupervised learning algorithms were compared:

1.  K-Means Clustering
2.  Agglomerative Hierarchical Clustering
3.  DBSCAN

The algorithms were evaluated using Silhouette Score, Davies-Bouldin
Index, and Calinski-Harabasz Index. Higher Silhouette and
Calinski-Harabasz values indicate better-separated clusters, while a
lower Davies-Bouldin value indicates better clustering.

## 4. Customer Segments

The clusters were interpreted using their RFM profiles:

-   Champions: Recent, frequent, high-spending customers who can receive
    loyalty rewards and exclusive offers.
-   High-Value Customers: Customers with comparatively high monetary
    value who can receive premium offers and cross-selling
    recommendations.
-   New Customers: Customers with relatively low purchase frequency or
    spending who can receive welcome offers and second-purchase
    incentives.
-   At-Risk Customers: Customers who have not purchased recently and can
    receive reactivation campaigns or limited-time discounts.

## 5. Business Recommendation

The clustering results provide a data-driven way to treat different
customer groups differently rather than using the same marketing
strategy for everyone. The final clustering approach should be selected
using the internal metric comparison together with business
interpretability. Customer personas can then be used to create targeted
campaigns.

## 6. Future Improvements

Future iterations could include product categories, average order value,
discount usage, customer demographics, marketing campaign responses,
browsing behaviour, payment method, acquisition channel, and
return/cancellation behaviour. Semi-supervised refinement could further
improve segmentation quality. The saved scaler and clustering model can
also be integrated into a real-time customer segmentation API.

## Conclusion

This project demonstrates an end-to-end customer segmentation workflow
using RFM analysis and unsupervised learning. Data cleaning, feature
engineering, outlier handling, transformation, scaling, clustering,
evaluation, visualization, and business interpretation were performed to
convert transaction data into actionable customer segments.
