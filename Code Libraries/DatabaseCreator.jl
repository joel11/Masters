#Pkg.add("SQLite")
using SQLite

function CreateDatabase(database_name)

    db = SQLite.DB(database_name)

    backtest_results_table = "CREATE TABLE IF NOT EXISTS backtest_results(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            stock VARCHAR,
                            actual FLOAT,
                            predicted FLOAT)"

    clusters_table = "CREATE TABLE IF NOT EXISTS clusters(
                        cluster INTEGER,
                        configuration_id INTEGER)"

    config_confusion_table = "CREATE TABLE IF NOT EXISTS config_confusion(
                        configuration_id INTEGER,
                        trades_percentage FLOAT,
                        no_trades_percentage FLOAT,
                        all_trades_percentage FLOAT)"

    config_is_pl = "CREATE TABLE IF NOT EXISTS config_is_pl(
                        configuration_id INTEGER,
                        total_pl FLOAT)"

    config_oos_pl = "CREATE TABLE IF NOT EXISTS config_oos_pl(
                        configuration_id INTEGER,
                        total_pl FLOAT)"

    config_oos_pl_cost = "CREATE TABLE IF NOT EXISTS config_oos_pl_cost(
                        configuration_id INTEGER,
                        total_pl FLOAT)"

    config_oos_sharpe_ratio = "CREATE TABLE IF NOT EXISTS config_oos_sharpe_ratio(
                                configuration_id INTEGER,
                                sharpe_ratio FLOAT)"

    config_oos_sharpe_ratio_cost = "CREATE TABLE IF NOT EXISTS config_oos_sharpe_ratio_cost(
                                configuration_id INTEGER,
                                sharpe_ratio FLOAT)"

    configuration_run_table = " CREATE TABLE IF NOT EXISTS configuration_run(
                                configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                experiment_set_name VARCHAR,
                                rbm_pretraining BOOL,
                                sae_config_id INTEGER,
                                seed_used INTEGER NOT NULL,
                                start_time DATETIME NOT NULL)"

    config_oos_trade_returns = "CREATE TABLE IF NOT EXISTS config_oos_trade_returns(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            total_profit_observed FLOAT,
                            total_profit_rate_observed FLOAT)"

    config_oos_trade_returns_cost = "CREATE TABLE IF NOT EXISTS config_oos_trade_returns_cost(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            total_profit_observed FLOAT,
                            total_profit_rate_observed FLOAT)"

    config_is_trade_returns = "CREATE TABLE IF NOT EXISTS config_is_trade_returns(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            total_profit_observed FLOAT,
                            total_profit_rate_observed FLOAT)"

    config_is_trade_returns_cost = "CREATE TABLE IF NOT EXISTS config_is_trade_returns_cost(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            total_profit_observed FLOAT,
                            total_profit_rate_observed FLOAT)"

    #=cscv_return_table = " CREATE TABLE IF NOT EXISTS cscv_returns(
                            configuration_id INTEGER,
                            time_step INTEGER,
                            total_profit_observed FLOAT,
                            total_profit_rate_observed FLOAT)"=#

    network_parameters_table = "
        CREATE TABLE IF NOT EXISTS network_parameters(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            layer_sizes VARCHAR NOT NULL,
            layer_activations VARCHAR NOT NULL,
            initialization VARCHAR NOT NULL,
            encoding_activation VARCHAR,
            output_activation VARCHAR)
            "

    training_parameters_table = "
        CREATE TABLE IF NOT EXISTS training_parameters(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            learning_rate FLOAT NOT NULL,
            minibatch_size INTEGER NOT NULL,
            max_epochs INTEGER NOT NULL,
            l1_lambda FLOAT,
            l2_lambda FLOAT,
            cost_function VARCHAR,
            stopping_function VARCHAR,
            min_learning_rate FLOAT,
            epoch_cycle_max INTEGER,
            is_denoising BOOL,
            denoising_variance FLOAT)"

    epoch_records_table = "
        CREATE TABLE IF NOT EXISTS epoch_records(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            record_time DATETIME NOT NULL,
            epoch_number INTEGER NOT NULL,
            mean_minibatch_cost FLOAT,
            training_cost FLOAT,
            testing_cost FLOAT,
            run_time FLOAT NOT NULL,
            learning_rate FLOAT)"

    dataset_config_table = "
        CREATE TABLE IF NOT EXISTS dataset_config(
            configuration_id INTEGER,
            data_seed INTEGER,
            category VARCHAR NOT NULL,
            steps INTEGER NOT NULL,
            deltas VARCHAR NOT NULL,
            process_splits VARCHAR,
            training_splits VARCHAR,
            prediction_steps VARCHAR NOT NULL,
            variation_values VARCHAR,
            scaling_function VARCHAR)"

    prediction_results_table = "
        CREATE TABLE IF NOT EXISTS prediction_results(
            configuration_id INTEGER,
            time_step INTEGER,
            stock VARCHAR,
            actual FLOAT,
            predicted FLOAT)"

    oos_trade_index = "CREATE INDEX IND_OOS_TRADE_CONFIGID ON config_oos_trade_returns(configuration_id);"
    oos_trade_cost_index = "CREATE INDEX IND_OOS_TRADE_COST_CONFIGID ON config_oos_trade_returns_cost(configuration_id);"

    is_trade_index = "CREATE INDEX IND_IS_TRADE_CONFIGID ON config_is_trade_returns(configuration_id);"
    is_trade_cost_index = "CREATE INDEX IND_IS_TRADE_COST_CONFIGID ON config_is_trade_returns_cost(configuration_id);"

    backtestindex = "CREATE INDEX IND_BACKTEST_RESULTS_CONFIGID ON backtest_results(configuration_id);"
    prediction_index = "CREATE INDEX IND_PREDICTION_RESULTS_CONFIGID ON prediction_results(configuration_id);"

    SQLite.execute!(db, backtest_results_table)
    SQLite.execute!(db, clusters_table)
    SQLite.execute!(db, config_confusion_table)
    SQLite.execute!(db, config_is_pl)
    SQLite.execute!(db, config_oos_pl)
    SQLite.execute!(db, config_oos_pl_cost)
    SQLite.execute!(db, config_oos_sharpe_ratio)
    SQLite.execute!(db, config_oos_sharpe_ratio_cost)
    SQLite.execute!(db, configuration_run_table)
    SQLite.execute!(db, training_parameters_table)
    SQLite.execute!(db, network_parameters_table)
    SQLite.execute!(db, epoch_records_table)
    SQLite.execute!(db, dataset_config_table)
    SQLite.execute!(db, prediction_results_table)
    SQLite.execute!(db, backtest_results_table)
    SQLite.execute!(db, config_is_trade_returns)
    SQLite.execute!(db, config_is_trade_returns_cost)
    SQLite.execute!(db, config_oos_trade_returns)
    SQLite.execute!(db, config_oos_trade_returns_cost)

    SQLite.execute!(db, oos_trade_index)
    SQLite.execute!(db, oos_trade_cost_index)
    SQLite.execute!(db, is_trade_index)
    SQLite.execute!(db, is_trade_cost_index)
    SQLite.execute!(db, backtestindex)
    SQLite.execute!(db, prediction_index)
end
