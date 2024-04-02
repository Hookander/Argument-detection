from models import *

def average_embeddings(hidden_states, attention_mask):
    """
    Args:
        hidden_states (torch.tensor): the hidden states from the model
        attention_mask (torch.tensor): the attention mask
    """
    # attention_mask has 1s where there are tokens and 0s where there is padding
    # we sum the hidden states and divide by the sum of the attention mask to get the average
    # we add a small value to the denominator to avoid division by zero
    return hidden_states.sum(1) / (attention_mask.sum(1).unsqueeze(-1) + 1e-5)

def show_tsne_dom(model, clean_props = True, typ = 'dom', ratio = [1., 0.]):
    sentences, cleaned_labels, domains = get_data_with_simp_labels()
    sentences = [sentences[i] for i in range(len(sentences)) if domains[i] != 0]
    domains = [d for d in domains if d != 0]
    print('len', len(sentences))
    tokenized_sentences = tokenize_sentences(sentences)

    train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, domains, ratio=ratio, batch_size=4)

    all_representations = torch.tensor([], device=device)
    i = 0
    with torch.no_grad():
        for tokenized_batch in train_dl:
            if i % 5 == 0:
                print(i)
            i += 1
            model_output = model(tokenized_batch)
            batch_representations = average_embeddings(model_output.logits, tokenized_batch["attention_mask"])
            all_representations = torch.cat((all_representations, batch_representations), 0)

    labels = domains
    train_labels = get_labels_from_ratio(labels, ratio)[0]
    if clean_props:
        for i in range(len(train_labels)):
            train_labels[i] = train_labels[i].strip()
            if train_labels[i][0] == 'P' and train_labels[i][1:].isdigit(): # Pn
                train_labels[i] = 'Pn'
    tsne = TSNE()
    print('ok')
    all_representations_2d = tsne.fit_transform(all_representations)
    #print(all_representations_2d.shape)
    scatter_plot = px.scatter(x=all_representations_2d[:, 0], y=all_representations_2d[:, 1], color=train_labels)
    scatter_plot.show(config={'staticPlot': True})
    #scatter_plot.write_html(f"tsne_{typ}.html")

#show_tsne_dom(Model("camembert-base", 21, lr = 1, weight_decay=0), typ='dom', clean_props = False)