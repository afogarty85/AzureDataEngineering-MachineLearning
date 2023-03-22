import numpy as np
import pandas as pd

# parent child recursion
def recur(df, init_parent, parents, parentChild=None, step=0, t_df=None):
    '''
    Recursively search and generate child / parent relationships

    params::
    df - DataFrame - the DF we are extracting nested data from
    init_parent - list - a single parent ID wrapped in a list
    parents - list - a list containing n-many new parents (childs of init_parent); to start, we use init_parent here
    parentChild - list - captures the parents of the downstream children
    step - int - capturing the depth of search
    t_df - DataFrame - storage container to yield; nothing to enter to start

    '''
    if len(parents) == 0:
        # end
        return

    if step >= 1:
        # give me the df
        yield t_df

    # generate some info of interest; set parent column to index for faster searching
    curr_pull = df[df.index.isin(parents)][["ID", "RELATED_ID", "SUBSTITUTES", "QUANTITY", "SORT_ORDER"]]
    # set the next parents
    nextParents = curr_pull.RELATED_ID.values
    # set current children
    currentChildren = curr_pull.RELATED_ID.values
    # the parents are set as the index
    currentParents = curr_pull.index.values
    # get quantities
    currentQuantities = curr_pull.QUANTITY.values
    # get subs
    currentSubstitutes = curr_pull.SUBSTITUTES.values
    # sortOrder
    currentSortOrder = curr_pull.SORT_ORDER.values

    # agg data
    t_df = pd.DataFrame({
                        'parentPartID': [init_parent] * len(nextParents),
                        'childPartID': currentChildren,
                        'childParentPartID': currentParents,
                        'Quantities': currentQuantities.astype(int),
                        'Substitutes': currentSubstitutes.astype(int),
                        'sortOrder': currentSortOrder.astype(int),
                        'Level': [step] * len(nextParents)
                        })

    # yield from
    yield from recur(df=df, init_parent=init_parent, parents=nextParents, parentChild=currentParents, step=step+1, t_df=t_df)


