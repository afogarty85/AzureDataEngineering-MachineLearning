import numpy as np
import pandas as pd

# parent child recursion
def recur(df, init_parent, parents, children, parentChild=None, step=0, t_df=None, nextExtras=None):
    '''
    Recursively search and generate child / parent relationships

    params::
    df - DataFrame - the DF we are extracting nested data from
    init_parent - list - a single parent ID wrapped in a list
    parents - list - a list containing n-many new parents (childs of init_parent); to start, we use init_parent here
    parentChild - list - captures the parents of the downstream children
    step - int - capturing the depth of search
    t_df - DataFrame - storage container to yield; nothing to enter to start
    nextExtras -- dict - conditionally generated item that will tell us to look for the right amount of new parent parts

    '''
    if len(parents) == 0:
        # end
        return

    if step >= 1:
        # give me the df
        yield t_df

    # generate some info of interest; set parent column to index for faster searching
    curr_pull = df[df.index.isin(parents)]

    if (nextExtras is not None) and (step > 1):
        # if we have duplicate children that are now parents, .isin will treat them uniquely, so
        for k, v in nextExtras.items():
            # extract set
            tdf_ = curr_pull[curr_pull.index.isin([k])]
            # multiply n times
            tdf_ = pd.concat(tdf_ for i in range(0, v-1))
            if len(tdf_) > 0:
                # add to current data
                curr_pull = pd.concat([curr_pull, tdf_], axis=0)

    # set the next parents
    nextParents = curr_pull[children].values
    # set current children
    currentChildren = curr_pull[children].values
    # the current parents are set as the index
    currentParents = curr_pull.index.values


    # prepare a dict to send into next 
    if curr_pull.groupby([children]).filter(lambda x: len(x) > 1).shape[0] > 0:
        sub_search = curr_pull.groupby([children]).filter(lambda x: len(x) > 1)
        nextExtras = sub_search.groupby([children]).size().to_dict()
    else:
        nextExtras = None

    # agg data
    t_df = pd.DataFrame({
                        'Parent': init_parent * len(nextParents),
                        'Child': currentChildren,
                        'childParent': currentParents,
                        'Level': [step] * len(nextParents)
                        })

    # yield from
    yield from recur(df=df, init_parent=init_parent, children=children, parents=nextParents, parentChild=currentParents, step=step+1, t_df=t_df, nextExtras=nextExtras)
