#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

raw_text = """<|endoftext|>"""

rawtext = """Tweet: We must repeal Obamacare and replace it with a much more competitive comprehensive affordable system.

Tweet: We're going to cut taxes BIG LEAGUE for the middle class. She's raising your taxes and I'm lowering yours!

Tweet: Remember when the failing @nytimes apologized to its subscribers right after the election because their coverage was so wrong. Now worse!

Tweet: The Fake News media is officially out of control. They will do or say anything in order to get attention - never been a time like this!

Tweet: Courageous Patriots have fought and died for our great American Flag --- we MUST honor and respect it! MAKE AMERICA GREAT AGAIN!

Tweet: A total WITCH HUNT with massive conflicts of interest!

Tweet: The Democrats are not doing what’s right for our country. I will not rest until we have secured our borders and restored the rule of law!

Tweet: The Fake News Media is going Crazy! They make up stories without any backup sources or proof. Many of the stories written about me and the good people surrounding me are total fiction. Problem is when you complain you just give them more publicity. But I’ll complain anyway!

Tweet: I am thrilled to announce that in the second quarter of this year the U.S. Economy grew at the amazing rate of 4.1%!

Tweet: We have accomplished an economic turnaround of HISTORIC proportions!

Tweet: Private business investment has surged from 1.8 percent the year BEFORE I came into office to 9.4 percent this year -- that means JOBS JOBS JOBS!

Tweet: Democrats who want Open Borders and care little about Crime are incompetent but they have the Fake News Media almost totally on their side!

Tweet: The only things the Democrats do well is “Resist” which is their campaign slogan and “Obstruct.” Cryin’ Chuck Schumer has almost 400 great American people that are waiting “forever” to serve our Country! A total disgrace. Mitch M should not let them go home until all approved!

Tweet: Wow highest Poll Numbers in the history of the Republican Party. That includes Honest Abe Lincoln and Ronald Reagan. There must be something wrong please recheck that poll!

Tweet: Do you think the Fake News Media will ever report on this tweet from Michael?

Tweet: Please understand there are consequences when people cross our Border illegally whether they have children or not - and many are just using children for their own sinister purposes. Congress must act on fixing the DUMBEST &amp; WORST immigration laws anywhere in the world! Vote “R”

Tweet: I would be willing to “shut down” government if the Democrats do not give us the votes for Border Security which includes the Wall! Must get rid of Lottery Catch &amp; Release etc. and finally go to system of Immigration based on MERIT! We need great people coming into our Country!

Tweet: There is No Collusion! The Robert Mueller Rigged Witch Hunt headed now by 17 (increased from 13 including an Obama White House lawyer) Angry Democrats was started by a fraudulent Dossier paid for by Crooked Hillary and the DNC. Therefore the Witch Hunt is an illegal Scam!

Tweet: MAKING AMERICA GREAT AGAIN!

Tweet: Rush Limbaugh is a great guy who truly gets it!

Tweet: Collusion is not a crime but that doesn’t matter because there was No Collusion (except by Crooked Hillary and the Democrats)!

Tweet: The Fake News Media is going CRAZY! They are totally unhinged and in many ways after witnessing first hand the damage they do to so many innocent and decent people I enjoy watching. In 7 years when I am no longer in office their ratings will dry up and they will be gone!

Tweet: Russian Collusion with the Trump Campaign one of the most successful in history is a TOTAL HOAX. The Democrats paid for the phony and discredited Dossier which was along with Comey McCabe Strzok and his lover the lovely Lisa Page used to begin the Witch Hunt. Disgraceful!

Tweet: “We already have a smocking gun about a campaign getting dirt on their opponent it was Hillary Clinton. How is it OK for Hillary Clinton to proactively seek dirt from the Russians but the Trump campaign met at the Russians request and that is bad?” Marc Thiessen Washington Post

Tweet: """

def interact_model(
    model_name='345M',
    seed=None,
    nsamples=10,
    batch_size=1,
    length=50,
    temperature=0.7,
    top_k=40,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(rawtext)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)
            

if __name__ == '__main__':
    fire.Fire(interact_model)