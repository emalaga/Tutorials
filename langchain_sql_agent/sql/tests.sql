







































-- What country has the biggest sales
SELECT 
	i.BillingCountry, 
	SUM(i.Total)
FROM 
	Invoice i 
GROUP BY 1
ORDER BY 2 DESC
LIMIT 5

























-- What album has sold the most tracks
SELECT 
	a.Title , 
	SUM(il.Quantity) as TrackSales
FROM 
	InvoiceLine il 
		INNER JOIN Track t ON
			il.TrackId = t.TrackId
		INNER JOIN Album a ON
			t.AlbumId = a.AlbumId 
GROUP BY 1
ORDER BY 2 DESC
LIMIT 5







-- Average duration of the top 5 best selling tracks
SELECT 
	AVG(t.Milliseconds) 
FROM 
	Track t 
WHERE 
	t.TrackId IN (
		SELECT 
			il.TrackId
		FROM 
			InvoiceLine il 
				INNER JOIN Track t ON
					il.TrackId = t.TrackId
		GROUP BY 1
		ORDER BY SUM(il.Quantity) DESC
		LIMIT 5
	)

	
	
	
	
-- What percent of customers have bought tracks from the 2 albums with the higest sales?
SELECT 
(
SELECT COUNT(DISTINCT i.CustomerId) 
FROM InvoiceLine il
JOIN Invoice i ON i.InvoiceId = il.InvoiceId 
JOIN Track t ON il.TrackId = t.TrackId
JOIN Album a ON t.AlbumId = a.AlbumId
WHERE 
	a.AlbumId IN (
	SELECT 
		t.AlbumId 
	FROM 
		InvoiceLine il 
			INNER JOIN Track t ON
				il.TrackId = t.TrackId
	GROUP BY 1
	ORDER BY SUM(il.UnitPrice * il.Quantity) DESC
	LIMIT 2
)
) / ( 
SELECT COUNT(DISTINCT CustomerId) * 1.0 FROM Invoice
) * 100

