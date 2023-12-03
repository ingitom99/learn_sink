"""
checkpoint.py
-------------

The function(s) to wrap together the evaluation phase of the training process
as well as the plotting and saving of results.
"""

from src.test_funcs import *
from src.plot import *

def checkpoint(gen_net, pred_net, test_sets, test_sinks, cost, eps, dim, device,
               results_folder, train_losses, test_losses, test_rel_errs_sink,
               test_mcvs, niter_warmstart):
    
    # Testing mode
    gen_net.eval()
    pred_net.eval()

    # Saving nets
    torch.save(gen_net.state_dict(), f'{results_folder}/deer.pt')
    torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')

    # Test warmstart
    warmstarts_sink = test_warmstart_sink(pred_net, test_sets, test_sinks,
                                cost, eps, dim, device, niter_warmstart)

    warmstarts_sink_1 = test_warmstart_sink_t(1, pred_net, test_sets, test_sinks,
                                cost, eps, dim, device)
    warmstarts_sink_5 = test_warmstart_sink_t(5, pred_net, test_sets, test_sinks,
                                cost, eps, dim, device)

    warmstarts_mcv = test_warmstart_MCV(pred_net, test_sets, cost, eps,
                                        dim, device, niter_warmstart)
    warmstarts_mcv_1 = test_warmstart_MCV_t(1, pred_net, test_sets, cost, eps,
                                        dim, device)
    warmstarts_mcv_5 = test_warmstart_MCV_t(5, pred_net, test_sets, cost, eps,
                                        dim, device)

    plot_train_losses(train_losses,
                                f'{results_folder}/train_losses.png')
    plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
    plot_test_rel_errs_sink(test_rel_errs_sink,
                            f'{results_folder}/test_rel_errs_sink.png')
    plot_test_mcvs(test_mcvs, f'{results_folder}/test_mcvs.png')
    plot_warmstarts_mcv(warmstarts_mcv, results_folder)
    plot_warmstarts_sink(warmstarts_sink, results_folder)
    plot_4(warmstarts_sink, f'{results_folder}/warmstarts_sink_grouped.png')
    plot_4(warmstarts_mcv, f'{results_folder}/warmstarts_mcv_grouped.png')
    plot_warmstart_violins(warmstarts_sink_1,
                            f'RE : iteration 1',
                            f'{results_folder}/violins_re_1.png')
    plot_warmstart_violins(warmstarts_sink_5,
                            'RE : iteration 5',
                            f'{results_folder}/violins_re_5.png')
    plot_warmstart_violins(warmstarts_mcv_1,
                            f'MCV : iteration 1',
                            f'{results_folder}/violins_mcv_1.png')
    plot_warmstart_violins(warmstarts_mcv_5,
                            f'MCV : iteration 5',
                            f'{results_folder}/violins_mcv_5.png')

    return (
        warmstarts_sink,
        warmstarts_mcv,
        warmstarts_sink_1,
        warmstarts_sink_5,
        warmstarts_mcv_1, 
        warmstarts_mcv_5
    )