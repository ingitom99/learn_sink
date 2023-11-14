from src.test_funcs import *
from src.plot import *

def checkpoint(gen_net, pred_net, test_sets, test_sinks, cost, eps, dim, device,
               results_folder, train_losses, test_losses, test_rel_errs_sink,
               test_mcvs, lipshitz_constants, niter_warmstart):
    
    # Testing mode
    gen_net.eval()
    pred_net.eval()

    # Saving nets
    torch.save(gen_net.state_dict(), f'{results_folder}/deer.pt')
    torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')

    # Test warmstart
    warmstarts_sink = test_warmstart_sink(pred_net, test_sets, test_sinks,
                                cost, eps, dim, device, niter_warmstart)

    warmstarts_sink_0 = test_warmstart_sink_t(0, pred_net, test_sets, test_sinks,
                                cost, eps, dim, device)
    warmstarts_sink_1 = test_warmstart_sink_t(1, pred_net, test_sets, test_sinks,
                                cost, eps, dim, device)
    # warmstarts_sink_5 = test_warmstart_sink_t(5, pred_net, test_sets, test_sinks,
    #                             cost, eps, dim, device)
    warmstarts_sink_10 = test_warmstart_sink_t(10, pred_net, test_sets, test_sinks,
                                cost, eps, dim, device)

    warmstarts_mcv = test_warmstart_MCV(pred_net, test_sets, cost, eps,
                                        dim, device, niter_warmstart)
    warmstarts_mcv_0 = test_warmstart_MCV_t(0, pred_net, test_sets, cost, eps,
                                        dim, device)
    warmstarts_mcv_1 = test_warmstart_MCV_t(1, pred_net, test_sets, cost, eps,
                                        dim, device)
    # warmstarts_mcv_5 = test_warmstart_MCV_t(5, pred_net, test_sets, cost, eps,
    #                                     dim, device)
    warmstarts_mcv_10 = test_warmstart_MCV_t(10, pred_net, test_sets, cost, eps,
                                        dim, device)

    plot_train_losses(train_losses,
                                f'{results_folder}/train_losses.png')
    plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
    plot_test_rel_errs_sink(test_rel_errs_sink,
                            f'{results_folder}/test_rel_errs_sink.png')
    plot_test_mcvs(test_mcvs, f'{results_folder}/test_mcvs.png')
    plot_warmstarts_mcv(warmstarts_mcv, results_folder)
    plot_warmstarts_sink(warmstarts_sink, results_folder)
    plot_warmstart_violins(warmstarts_sink_0,
                           r'$E_{SD} : iteration 0',
                           f'{results_folder}/violins_rel_err_sink_0.png')
    plot_warmstart_violins(warmstarts_sink_1,
                            r'$E_{SD} : iteration 1',
                            f'{results_folder}/violins_rel_err_sink_1.png')
    plot_warmstart_violins(warmstarts_sink_10,
                            r'$E_{SD} : iteration 10',
                            f'{results_folder}/violins_rel_err_sink_10.png')
    plot_warmstart_violins(warmstarts_mcv_0,
                            r'$E_{MCV} : iteration 0',
                            f'{results_folder}/violins_mcv_0.png')
    plot_warmstart_violins(warmstarts_mcv_1,
                            r'$E_{MCV} : iteration 1',
                            f'{results_folder}/violins_mcv_1.png')
    plot_warmstart_violins(warmstarts_mcv_10,
                            r'$E_{MCV} : iteration 10',
                            f'{results_folder}/violins_mcv_10.png')
    plot_lipschitz_vals(lipshitz_constants,
                                f'{results_folder}/lipschitz_vals.png')

    return (
        warmstarts_sink,
        warmstarts_mcv,
        warmstarts_sink_0,
        warmstarts_sink_1,
        warmstarts_sink_10,
        warmstarts_mcv_0,
        warmstarts_mcv_1, 
        warmstarts_mcv_10
    )