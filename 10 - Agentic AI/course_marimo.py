import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(path):
    path
    return


@app.cell
def _():
    import pandas as pd
    import os

    return os, pd


@app.cell
def _(os, path, pd):
    books = pd.read_csv(os.path.join(path, 'books.csv'))
    return (books,)


@app.cell
def _(books):
    books.head()
    return


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt

    return plt, sns


@app.cell
def _(books, plt, sns):
    ax = plt.axes()
    sns.heatmap(books.isna().transpose(),cbar = True, ax=ax)

    plt.show()
    return


@app.cell
def _(books):
    book_missing = books[~(books["description"].isna()) &
          ~(books["num_pages"].isna()) &
          ~(books["average_rating"].isna()) &
          ~(books["published_year"].isna())
    ]
    return (book_missing,)


@app.cell
def _(book_missing):
    book_missing
    return


@app.cell
def _(book_missing):
    book_missing['categories'].value_counts().reset_index().sort_values('count', ascending=False)
    return


@app.cell
def _(book_missing):
    book_missing['words_in_description'] = book_missing['description'].str.split().str.len()
    return


@app.cell
def _(book_missing):
    book_missing['words_in_description']
    return


@app.cell
def _(book_missing, sns):
    sns.histplot(x='words_in_description', data = book_missing)
    return


@app.cell
def _(book_missing):
    book_missing.loc[book_missing['words_in_description'].between(1, 4), 'description']
    return


@app.cell
def _(book_missing):
    book_missing_25_words = book_missing[book_missing['words_in_description'] >= 25]
    return (book_missing_25_words,)


@app.cell
def _(book_missing_25_words):
    book_missing_25_words.shape
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(book_missing_25_words, np):
    book_missing_25_words['title_and_subtitle'] = (
        np.where(book_missing_25_words['subtitle'].isna(), book_missing_25_words['title'], 
                # f"{book_missing_25_words['title']:{book_missing_25_words['subtitle']}")
                 book_missing_25_words[['title', 'subtitle']].astype(str).agg(":".join, axis = 1))
    )
    return


@app.cell
def _(book_missing_25_words):
    book_missing_25_words['title_and_subtitle']
    return


@app.cell
def _(book_missing_25_words):
    book_missing_25_words['tagged_description'] = book_missing_25_words[['isbn13', 'description']].astype(str).agg(" ".join, axis = 1)
    return


@app.cell
def _(book_missing_25_words):
    book_missing_25_words['tagged_description']
    return


@app.cell
def _(book_missing_25_words):
    (
        book_missing_25_words.drop([
            'subtitle',
            # 'missing_description',
            'words_in_description'
        ], axis = 1)
        .to_csv('./data/books_cleaned.csv', index = False)
    )
    return


if __name__ == "__main__":
    app.run()
